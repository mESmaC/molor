"""JSONL chat examples and assistant-only causal-LM batching."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("message content must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class ChatExample:
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("an example must contain messages")
        if self.messages[-1].role != "assistant":
            raise ValueError("the final message must be the assistant response to train")
        if not any(message.role == "user" for message in self.messages[:-1]):
            raise ValueError("an example needs a user message before the assistant response")

    @classmethod
    def from_dict(cls, value: object) -> ChatExample:
        if not isinstance(value, dict) or set(value) != {"messages"}:
            raise ValueError("each JSONL row must contain only a messages array")
        raw_messages = value["messages"]
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be an array")
        messages: list[ChatMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
                raise ValueError("each message needs exactly role and content")
            messages.append(ChatMessage(role=raw["role"], content=raw["content"]))
        return cls(tuple(messages))


@dataclass(frozen=True, slots=True)
class EncodedChatExample:
    """One rendered chat with assistant-only labels, before batch padding."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.labels):
            raise ValueError("encoded input IDs and labels need the same non-zero length")
        if any(type(token_id) is not int or token_id < 0 for token_id in self.input_ids):
            raise ValueError("encoded input IDs must be non-negative integers")
        if any(type(label) is not int or (label != -100 and label < 0) for label in self.labels):
            raise ValueError("encoded labels must be token IDs or -100")
        if any(
            label != -100 and label != token_id
            for token_id, label in zip(self.input_ids, self.labels, strict=True)
        ):
            raise ValueError("unmasked labels must match their input token IDs")
        if all(label == -100 for label in self.labels):
            raise ValueError("encoded example must contain an assistant target")


def load_jsonl(path: str | Path) -> list[ChatExample]:
    """Load the intentionally tiny training format used by the toy experiments."""

    source = Path(path)
    examples: list[ChatExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(ChatExample.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"{source} contains no examples")
    return examples


class ChatTemplateTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: Any,
    ) -> object: ...


class AssistantOnlyCollator:
    """Render chats and mask every token except the final assistant response.

    This uses the tokenizer's own generation prefix rather than guessing special-token IDs.
    Examples that exceed ``max_length`` are cropped from the left, preserving the answer and the
    most recent part of its prompt.
    """

    def __init__(
        self,
        tokenizer: ChatTemplateTokenizer,
        *,
        max_length: int,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if any(
            key in self.chat_template_kwargs
            for key in (
                "tokenize",
                "add_generation_prompt",
                "return_tensors",
                "return_dict",
                "padding",
                "truncation",
            )
        ):
            raise ValueError("chat_template_kwargs cannot override collation controls")
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer needs a pad_token_id or eos_token_id")

    @staticmethod
    def _token_ids(value: object) -> list[int]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError("chat template must return a non-empty flat token list")
        if any(type(token_id) is not int or token_id < 0 for token_id in value):
            raise ValueError("chat template returned invalid token IDs")
        return list(value)

    def _render(self, messages: tuple[ChatMessage, ...], *, generating: bool) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [message.to_dict() for message in messages],
            tokenize=True,
            add_generation_prompt=generating,
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        return self._token_ids(rendered)

    def encode(self, example: ChatExample) -> EncodedChatExample:
        """Render one example once so it can be reused across epochs or regrouped."""

        prompt_ids = self._render(example.messages[:-1], generating=True)
        full_ids = self._render(example.messages, generating=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "tokenizer chat template has inconsistent training and generation prefixes"
            )

        prompt_length = len(prompt_ids)
        if len(full_ids) > self.max_length:
            cropped = len(full_ids) - self.max_length
            full_ids = full_ids[cropped:]
            prompt_length = max(0, prompt_length - cropped)
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if all(label == -100 for label in labels):
            raise ValueError("max_length removed the entire assistant response")
        return EncodedChatExample(tuple(full_ids), tuple(labels))

    def collate_encoded(self, examples: list[EncodedChatExample]) -> dict[str, Tensor]:
        """Pad examples that were already rendered by :meth:`encode`."""

        if not examples:
            raise ValueError("cannot collate an empty batch")
        width = max(len(example.input_ids) for example in examples)
        input_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for example in examples:
            input_ids = list(example.input_ids)
            labels = list(example.labels)
            padding = width - len(input_ids)
            input_rows.append(input_ids + [self.pad_token_id] * padding)
            label_rows.append(labels + [-100] * padding)
            attention_rows.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }

    def __call__(self, examples: list[ChatExample]) -> dict[str, Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        return self.collate_encoded([self.encode(example) for example in examples])
