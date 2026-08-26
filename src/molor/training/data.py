"""JSONL chat examples and assistant-only causal-LM batching."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains unsupported numeric constant {value!r}")


def _strict_json_loads(value: str, field_name: str) -> object:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ValueError(f"{field_name} must be valid strict JSON: {exc}") from exc


def _validate_unicode_scalar_string(value: str, field_name: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{field_name} must contain only Unicode scalar values")


def _validate_json_value(value: object, field_name: str) -> None:
    if isinstance(value, str):
        _validate_unicode_scalar_string(value, field_name)
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} contains a non-string object key")
            _validate_json_value(item, f"{field_name}.{key}")
        return
    raise ValueError(f"{field_name} contains a non-JSON value")


class ToolTraceError(ValueError):
    """A structured tool trace that fails one stable, aggregate-safe contract category."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ToolFunction:
    name: str
    arguments_json: str
    _arguments_encoded: bool = field(default=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool function name must be a non-empty string")
        _validate_unicode_scalar_string(self.name, "tool function name")
        if not isinstance(self.arguments_json, str):
            raise TypeError("tool function arguments_json must be a string")
        if type(self._arguments_encoded) is not bool:
            raise TypeError("_arguments_encoded must be a boolean")
        arguments = _strict_json_loads(self.arguments_json, "tool function arguments")
        if not isinstance(arguments, dict):
            raise ValueError("tool function arguments must encode one JSON object")
        _validate_json_value(arguments, "tool function arguments")

    @classmethod
    def from_dict(cls, value: object) -> ToolFunction:
        if not isinstance(value, Mapping) or set(value) != {"name", "arguments"}:
            raise ValueError("tool function needs exactly name and arguments")
        arguments = value["arguments"]
        if isinstance(arguments, str):
            arguments_json = arguments
            arguments_encoded = True
        elif isinstance(arguments, dict):
            _validate_json_value(arguments, "tool function arguments")
            arguments_json = json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            arguments_encoded = False
        else:
            raise ValueError("tool function arguments must be a JSON object or encoded object")
        return cls(
            name=value["name"],  # type: ignore[arg-type]
            arguments_json=arguments_json,
            _arguments_encoded=arguments_encoded,
        )

    def to_dict(self) -> dict[str, object]:
        arguments: object
        if self._arguments_encoded:
            arguments = self.arguments_json
        else:
            arguments = _strict_json_loads(self.arguments_json, "tool function arguments")
        return {"name": self.name, "arguments": arguments}


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str | None
    function: ToolFunction
    call_type: str = "function"
    _call_id_present: bool = field(default=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.call_id is not None and (
            not isinstance(self.call_id, str) or not self.call_id.strip()
        ):
            raise ValueError("tool call id must be a non-empty string or None")
        if self.call_id is not None:
            _validate_unicode_scalar_string(self.call_id, "tool call id")
        if not isinstance(self.function, ToolFunction):
            raise TypeError("tool call function must be a ToolFunction")
        if self.call_type != "function":
            raise ValueError("only function tool calls are supported")
        if type(self._call_id_present) is not bool:
            raise TypeError("_call_id_present must be a boolean")
        if self.call_id is not None:
            object.__setattr__(self, "_call_id_present", True)

    @classmethod
    def from_dict(cls, value: object) -> ToolCall:
        if not isinstance(value, Mapping) or set(value) not in (
            {"type", "function"},
            {"id", "type", "function"},
        ):
            raise ValueError("tool call needs type/function and an optional id")
        return cls(
            call_id=value.get("id"),  # type: ignore[arg-type]
            call_type=value["type"],  # type: ignore[arg-type]
            function=ToolFunction.from_dict(value["function"]),
            _call_id_present="id" in value,
        )

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "type": self.call_type,
            "function": self.function.to_dict(),
        }
        if self._call_id_present:
            value["id"] = self.call_id
        return value


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    _tool_calls_present: bool = field(default=False, repr=False, kw_only=True)
    _tool_call_id_present: bool = field(default=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat role: {self.role!r}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        _validate_unicode_scalar_string(self.content, "message content")
        if not self.content.strip() and not (
            self.role == "tool" or (self.role == "assistant" and self.tool_calls)
        ):
            raise ValueError("message content must be a non-empty string")
        if not isinstance(self.tool_calls, tuple) or any(
            not isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise TypeError("tool_calls must be a tuple of ToolCall values")
        if type(self._tool_calls_present) is not bool:
            raise TypeError("_tool_calls_present must be a boolean")
        if type(self._tool_call_id_present) is not bool:
            raise TypeError("_tool_call_id_present must be a boolean")
        if self.tool_calls:
            object.__setattr__(self, "_tool_calls_present", True)
        if self._tool_calls_present and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool_calls")
        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
        ):
            raise ValueError("tool_call_id must be a non-empty string or None")
        if self.tool_call_id is not None:
            _validate_unicode_scalar_string(self.tool_call_id, "tool_call_id")
        if self.tool_call_id is not None:
            object.__setattr__(self, "_tool_call_id_present", True)
        if self.role != "tool" and self._tool_call_id_present:
            raise ValueError("only tool messages may contain tool_call_id")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {"role": self.role, "content": self.content}
        if self._tool_calls_present:
            value["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        if self._tool_call_id_present:
            value["tool_call_id"] = self.tool_call_id
        return value

    @classmethod
    def from_dict(cls, value: object) -> ChatMessage:
        if not isinstance(value, Mapping):
            raise ValueError("each message must be an object")
        allowed = {"role", "content", "tool_calls", "tool_call_id"}
        if not {"role", "content"}.issubset(value) or not set(value).issubset(allowed):
            raise ValueError(
                "each message needs role/content and only optional tool_calls/tool_call_id"
            )
        raw_calls = value.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("tool_calls must be an array")
        return cls(
            role=value["role"],  # type: ignore[arg-type]
            content=value["content"],  # type: ignore[arg-type]
            tool_calls=tuple(ToolCall.from_dict(call) for call in raw_calls),
            tool_call_id=value.get("tool_call_id"),  # type: ignore[arg-type]
            _tool_calls_present="tool_calls" in value,
            _tool_call_id_present="tool_call_id" in value,
        )


@dataclass(frozen=True, slots=True)
class ChatExample:
    messages: tuple[ChatMessage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple) or any(
            not isinstance(message, ChatMessage) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of ChatMessage values")
        if not self.messages:
            raise ValueError("an example must contain messages")
        if self.messages[-1].role != "assistant":
            raise ValueError("the final message must be the assistant response to train")
        if not any(message.role == "user" for message in self.messages[:-1]):
            raise ValueError("an example needs a user message before the assistant response")
        self._validate_tool_trace()

    def _validate_tool_trace(self) -> None:
        """Validate linkage while admitting one pre-assistant truncated-observation window."""

        pending_by_id: dict[str, ToolCall] | None = None
        pending_without_ids = 0
        seen_call_ids: set[str] = set()
        orphan_result_ids: set[str] = set()
        initial_orphan_window_open = True
        orphan_block_started = False

        for index, message in enumerate(self.messages):
            if pending_by_id is not None or pending_without_ids:
                if message.role != "tool":
                    raise ToolTraceError(
                        "tool_result_count_mismatch",
                        "assistant tool calls must be followed by exactly one tool result per call",
                    )
                if pending_by_id is not None:
                    if message.tool_call_id is None:
                        raise ToolTraceError(
                            "missing_tool_call_id",
                            "tool result is missing its pending tool_call_id",
                        )
                    if message.tool_call_id not in pending_by_id:
                        raise ToolTraceError(
                            "unknown_or_duplicate_tool_call_id",
                            f"tool result references unknown or duplicate call ID "
                            f"{message.tool_call_id!r}",
                        )
                    del pending_by_id[message.tool_call_id]
                    if not pending_by_id:
                        pending_by_id = None
                else:
                    if message.tool_call_id is not None:
                        raise ToolTraceError(
                            "idless_call_result_mismatch",
                            "ID-less tool calls require ID-less tool results",
                        )
                    pending_without_ids -= 1
                continue

            if message.role == "tool":
                if not initial_orphan_window_open:
                    raise ToolTraceError(
                        "orphan_tool_result",
                        "tool result has no preceding assistant tool call",
                    )
                orphan_block_started = True
                if message.tool_call_id is not None:
                    if message.tool_call_id in orphan_result_ids:
                        raise ToolTraceError(
                            "duplicate_orphan_tool_call_id",
                            f"orphan tool-result prefix repeats call ID {message.tool_call_id!r}",
                        )
                    orphan_result_ids.add(message.tool_call_id)
                    seen_call_ids.add(message.tool_call_id)
                continue

            if orphan_block_started and message.role != "assistant":
                raise ToolTraceError(
                    "orphan_observation_window_not_closed",
                    "an initial orphan tool-result block must be followed by an assistant",
                )
            if message.role == "assistant":
                initial_orphan_window_open = False
            if not message.tool_calls:
                continue

            call_ids = [call.call_id for call in message.tool_calls]
            populated_ids = [call_id for call_id in call_ids if call_id is not None]
            if populated_ids and len(populated_ids) != len(call_ids):
                raise ToolTraceError(
                    "mixed_tool_call_id_mode",
                    "one assistant tool-call batch cannot mix ID-bearing and ID-less calls",
                )
            if len(populated_ids) != len(set(populated_ids)):
                raise ToolTraceError(
                    "duplicate_tool_call_id",
                    "assistant tool-call IDs must be unique within a call batch",
                )
            repeated = seen_call_ids.intersection(populated_ids)
            if repeated:
                raise ToolTraceError(
                    "reused_tool_call_id",
                    f"assistant tool-call ID is reused: {min(repeated)!r}",
                )
            seen_call_ids.update(populated_ids)

            if index == len(self.messages) - 1:
                continue
            if populated_ids:
                pending_by_id = {
                    call.call_id: call for call in message.tool_calls if call.call_id is not None
                }
            else:
                pending_without_ids = len(message.tool_calls)

    @classmethod
    def from_dict(cls, value: object) -> ChatExample:
        if not isinstance(value, dict) or set(value) != {"messages"}:
            raise ValueError("each JSONL row must contain only a messages array")
        raw_messages = value["messages"]
        if not isinstance(raw_messages, list):
            raise ValueError("messages must be an array")
        messages: list[ChatMessage] = []
        for raw in raw_messages:
            messages.append(ChatMessage.from_dict(raw))
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
                raise ValueError(f"{source}:{line_number}: blank JSONL records are not allowed")
            try:
                value = _strict_json_loads(line, "JSONL record")
                examples.append(ChatExample.from_dict(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"{source} contains no examples")
    return examples


class ChatTemplateTokenizer(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
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
                "continue_final_message",
                "return_assistant_tokens_mask",
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
