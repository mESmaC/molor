"""Deterministic tokenizer and generation backend for local smoke testing.

These primitives exercise the real prompt, execution-plan, cancellation, and stream
boundaries without loading model libraries or tensors. They are deliberately static:
the backend response is immutable configuration rather than prompt-dependent behavior.
"""

from __future__ import annotations

import asyncio
import codecs
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

from molor.identity import canonical_json_bytes, fingerprint_json
from molor.runtime.execution import (
    BackendDelta,
    BackendDescriptor,
    BackendFinished,
    BackendProgress,
    BackendRecord,
    CancellationToken,
    ExecutionPlan,
    PromptMessage,
)


_MAX_CONTEXT_TOKENS = 4_194_304
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_CHUNK_TOKENS = 8_192


def _positive_int(value: int, field_name: str, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum:
        raise ValueError(f"{field_name} must be between 1 and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class ByteRuntimeTokenizer:
    """Canonical chat serialization where each UTF-8 byte is one token ID."""

    schema_version: ClassVar[str] = "molor.byte-runtime-tokenizer.v1"

    context_limit: int = 131_072
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        context_limit = _positive_int(
            self.context_limit,
            "context_limit",
            maximum=_MAX_CONTEXT_TOKENS,
        )
        object.__setattr__(self, "context_limit", context_limit)
        object.__setattr__(
            self,
            "_fingerprint",
            fingerprint_json(
                {
                    "schema_version": self.schema_version,
                    "encoding": "utf-8-byte-token-ids",
                    "chat_serialization": "canonical-json",
                    "chat_schema_version": "molor.byte-chat-prompt.v1",
                    "context_limit": context_limit,
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return len(text.encode("utf-8"))

    def encode_text(self, text: str) -> tuple[int, ...]:
        """Encode arbitrary text into exact byte token IDs."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return tuple(text.encode("utf-8"))

    def decode_tokens(self, token_ids: tuple[int, ...]) -> str:
        """Decode one complete valid UTF-8 byte-token sequence."""

        tokens = tuple(token_ids)
        if any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or not 0 <= token_id <= 255
            for token_id in tokens
        ):
            raise ValueError("byte token IDs must be integers in [0, 255]")
        return bytes(tokens).decode("utf-8", errors="strict")

    def encode_chat(
        self,
        messages: tuple[PromptMessage, ...],
        *,
        add_generation_prompt: bool,
    ) -> tuple[int, ...]:
        if not isinstance(add_generation_prompt, bool):
            raise TypeError("add_generation_prompt must be boolean")
        messages = tuple(messages)
        if not messages:
            raise ValueError("messages must not be empty")
        if any(not isinstance(message, PromptMessage) for message in messages):
            raise TypeError("messages must contain only PromptMessage values")
        serialized = canonical_json_bytes(
            {
                "schema_version": "molor.byte-chat-prompt.v1",
                "messages": [
                    {"role": message.role, "content": message.content} for message in messages
                ],
                "generation_prompt": ({"role": "assistant"} if add_generation_prompt else None),
            }
        )
        return tuple(serialized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context_limit": self.context_limit,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class DeterministicGenerationBackend:
    """Emit one immutable UTF-8 response through the real backend protocol."""

    schema_version: ClassVar[str] = "molor.deterministic-generation-backend.v1"

    descriptor: BackendDescriptor
    response_text: str = "molor deterministic runtime is ready."
    chunk_tokens: int = 16
    _response_bytes: bytes = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, BackendDescriptor):
            raise TypeError("descriptor must be a BackendDescriptor")
        if not isinstance(self.response_text, str) or not self.response_text:
            raise ValueError("response_text must be a non-empty string")
        if any(
            ord(character) < 32 and character not in "\n\r\t" for character in self.response_text
        ):
            raise ValueError("response_text contains an unsupported control character")
        response_bytes = self.response_text.encode("utf-8")
        if len(response_bytes) > _MAX_RESPONSE_BYTES:
            raise ValueError("response_text exceeds the deterministic response byte limit")
        chunk_tokens = _positive_int(
            self.chunk_tokens,
            "chunk_tokens",
            maximum=_MAX_CHUNK_TOKENS,
        )
        object.__setattr__(self, "chunk_tokens", chunk_tokens)
        object.__setattr__(self, "_response_bytes", response_bytes)
        object.__setattr__(
            self,
            "_fingerprint",
            fingerprint_json(
                {
                    "schema_version": self.schema_version,
                    "descriptor_fingerprint": self.descriptor.fingerprint,
                    "response_sha256": fingerprint_json(
                        {
                            "schema_version": "molor.deterministic-response.v1",
                            "text": self.response_text,
                        }
                    ),
                    "response_tokens": len(response_bytes),
                    "chunk_tokens": chunk_tokens,
                    "progress_phases": ["loading", "prefill", "decoding"],
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    async def generate(
        self,
        plan: ExecutionPlan,
        cancellation: CancellationToken,
    ) -> AsyncIterator[BackendRecord]:
        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be an ExecutionPlan")
        if not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken")
        if plan.backend != self.descriptor:
            raise ValueError("execution plan targets a different backend descriptor")

        for phase in ("loading", "prefill", "decoding"):
            await self._checkpoint(cancellation)
            yield BackendProgress(phase)

        max_tokens = plan.decoding.max_new_tokens
        emitted_bytes = self._response_bytes[:max_tokens]
        truncated = len(self._response_bytes) > max_tokens
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        for offset in range(0, len(emitted_bytes), self.chunk_tokens):
            await self._checkpoint(cancellation)
            chunk = emitted_bytes[offset : offset + self.chunk_tokens]
            text = decoder.decode(chunk, final=False)
            yield BackendDelta(text=text, token_ids=tuple(chunk))

        if not truncated:
            tail = decoder.decode(b"", final=True)
            if tail:
                # Complete UTF-8 input is emitted eagerly by Python's incremental
                # decoder, so a non-empty final tail would violate this backend's
                # token/text association contract.
                raise RuntimeError("byte decoder retained unexpected final text")
        await self._checkpoint(cancellation)
        yield BackendFinished("length" if truncated else "stop")

    @staticmethod
    async def _checkpoint(cancellation: CancellationToken) -> None:
        cancellation.raise_if_cancelled()
        await asyncio.sleep(0)
        cancellation.raise_if_cancelled()
