from __future__ import annotations

import torch

import pytest

from molor.training import (
    AssistantOnlyCollator,
    ChatExample,
    ChatMessage,
    ToolCall,
    ToolFunction,
    load_jsonl,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        **kwargs: object,
    ) -> list[int]:
        prompt = [1, 2, 3]
        if kwargs["add_generation_prompt"]:
            return prompt
        return prompt + [len(conversation[-1]["content"]), self.eos_token_id]


class CapturingTokenizer(TinyTokenizer):
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, object]]] = []

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> list[int]:
        self.conversations.append(conversation)
        prompt = [1, 2, 3]
        if kwargs["add_generation_prompt"]:
            return prompt
        return prompt + [7, self.eos_token_id]


class QwenStyleTokenizer:
    """Small deterministic stand-in for Qwen3's shared assistant generation prefix."""

    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        conversation: list[dict[str, object]],
        **kwargs: object,
    ) -> list[int]:
        enable_thinking = kwargs.get("enable_thinking", True)
        assistant_prefix = [50] if enable_thinking else [50, 51, 52]
        result = [1]
        for message in conversation:
            role = message["role"]
            if role == "user":
                result.extend((10, len(str(message["content"]))))
            elif role == "tool":
                result.extend((20, len(str(message["content"])), self.eos_token_id))
            elif role == "assistant":
                result.extend(assistant_prefix)
                calls = message.get("tool_calls")
                if calls:
                    assert isinstance(calls, list)
                    result.append(70 + len(calls))
                else:
                    result.append(60 + len(str(message["content"])))
                result.append(self.eos_token_id)
            else:
                result.extend((30, len(str(message["content"]))))
        if kwargs["add_generation_prompt"]:
            result.extend(assistant_prefix)
        return result


def test_collator_masks_prompt_and_padding_but_learns_assistant() -> None:
    examples = [
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "short"))),
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "longer"))),
    ]

    batch = AssistantOnlyCollator(TinyTokenizer(), max_length=16)(examples)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 5, 9], [1, 2, 3, 6, 9]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 5, 9], [-100, -100, -100, 6, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]


def test_encoded_collation_exactly_matches_online_collation() -> None:
    examples = [
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "short"))),
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "longer"))),
    ]
    collator = AssistantOnlyCollator(TinyTokenizer(), max_length=16)

    online = collator(examples)
    encoded = collator.collate_encoded([collator.encode(example) for example in examples])

    assert set(online) == set(encoded)
    for key in online:
        assert torch.equal(online[key], encoded[key])


def test_structured_tool_trace_round_trips_into_chat_template() -> None:
    raw = {
        "messages": [
            {"role": "user", "content": "Look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"id": 7}},
                    }
                ],
            },
            {"role": "tool", "content": "found", "tool_call_id": "call-1"},
            {"role": "assistant", "content": "The record was found."},
        ]
    }
    example = ChatExample.from_dict(raw)
    tokenizer = CapturingTokenizer()

    encoded = AssistantOnlyCollator(tokenizer, max_length=32).encode(example)

    assert encoded.labels[-2:] == (7, tokenizer.eos_token_id)
    prompt_messages = tokenizer.conversations[0]
    assert prompt_messages[1] == raw["messages"][1]
    assert prompt_messages[2] == raw["messages"][2]
    assert example.messages[1].tool_calls[0].function.arguments_json == '{"id":7}'


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen_style_prefix_masks_prior_tool_trace_and_trains_final_answer(
    enable_thinking: bool,
) -> None:
    example = ChatExample.from_dict(
        {
            "messages": [
                {"role": "user", "content": "Look it up"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"id": 7}},
                        }
                    ],
                },
                {"role": "tool", "content": "found", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "The record was found."},
            ]
        }
    )
    encoded = AssistantOnlyCollator(
        QwenStyleTokenizer(),
        max_length=64,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    ).encode(example)

    expected_target = (60 + len("The record was found."), QwenStyleTokenizer.eos_token_id)
    assert encoded.labels[-2:] == expected_target
    assert all(label == -100 for label in encoded.labels[:-2])


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen_style_prefix_trains_a_final_native_tool_call(enable_thinking: bool) -> None:
    example = ChatExample.from_dict(
        {
            "messages": [
                {"role": "user", "content": "Look it up"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{  "id" : 7  }',
                            },
                        }
                    ],
                },
            ]
        }
    )
    encoded = AssistantOnlyCollator(
        QwenStyleTokenizer(),
        max_length=64,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    ).encode(example)

    assert encoded.labels[-2:] == (71, QwenStyleTokenizer.eos_token_id)
    assert all(label == -100 for label in encoded.labels[:-2])


def test_native_tool_schema_round_trips_string_object_null_empty_and_absent_fields() -> None:
    raw = {
        "messages": [
            {"role": "user", "content": "Look it up"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{  "id" : 7  }',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "", "tool_call_id": None},
            {"role": "assistant", "content": "done", "tool_calls": []},
        ]
    }
    example = ChatExample.from_dict(raw)

    assert [message.to_dict() for message in example.messages] == raw["messages"]
    absent_id = ToolCall.from_dict(
        {
            "type": "function",
            "function": {"name": "lookup", "arguments": {"id": 7}},
        }
    )
    assert absent_id.to_dict() == {
        "type": "function",
        "function": {"name": "lookup", "arguments": {"id": 7}},
    }


def test_initial_tool_results_are_an_explicit_truncated_observation_window() -> None:
    raw = {
        "messages": [
            {"role": "system", "content": "Use prior observations"},
            {"role": "user", "content": "Summarize those results"},
            {"role": "tool", "content": "first", "tool_call_id": "old-1"},
            {"role": "tool", "content": "second", "tool_call_id": "old-2"},
            {"role": "assistant", "content": "summary"},
        ]
    }

    example = ChatExample.from_dict(raw)
    assert [message.to_dict() for message in example.messages] == raw["messages"]
    AssistantOnlyCollator(QwenStyleTokenizer(), max_length=64).encode(example)


@pytest.mark.parametrize(
    ("messages", "match"),
    [
        (
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "prior answer"},
                {"role": "tool", "content": "orphan", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "answer"},
            ],
            "no preceding assistant tool call",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "wrong"},
                {"role": "assistant", "content": "answer"},
            ],
            "unknown or duplicate call ID",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {}},
                        }
                    ],
                },
                {"role": "assistant", "content": "answer"},
            ],
            "exactly one tool result",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "one", "arguments": {}},
                        },
                        {
                            "type": "function",
                            "function": {"name": "two", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "content": "one", "tool_call_id": "call-1"},
                {"role": "tool", "content": "two"},
                {"role": "assistant", "content": "answer"},
            ],
            "cannot mix ID-bearing and ID-less",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "one", "arguments": {}},
                        },
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "two", "arguments": {}},
                        },
                    ],
                },
                {"role": "tool", "content": "one", "tool_call_id": "call-1"},
                {"role": "tool", "content": "two", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "answer"},
            ],
            "must be unique",
        ),
        (
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "one", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "one", "tool_call_id": "call-1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "two", "arguments": {}},
                        }
                    ],
                },
                {"role": "tool", "content": "two", "tool_call_id": "call-1"},
                {"role": "assistant", "content": "answer"},
            ],
            "is reused",
        ),
        (
            [
                {"role": "system", "content": "Use prior observations"},
                {"role": "user", "content": "question"},
                {"role": "tool", "content": "old", "tool_call_id": "call-1"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {}},
                        }
                    ],
                },
            ],
            "is reused",
        ),
    ],
)
def test_tool_trace_linkage_fails_closed(messages: list[dict[str, object]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ChatExample.from_dict({"messages": messages})


def test_strict_tool_json_and_jsonl_reject_duplicates_nonfinite_and_blank_records(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="valid strict JSON"):
        ToolFunction("lookup", '{"id":1,"id":2}')
    with pytest.raises(ValueError, match="valid strict JSON"):
        ToolFunction("lookup", '{"id":NaN}')
    with pytest.raises(ValueError, match="Unicode scalar"):
        ToolFunction("lookup", '{"id":"\\ud800"}')
    with pytest.raises(ValueError, match="non-string object key"):
        ToolFunction.from_dict({"name": "lookup", "arguments": {1: "value"}})

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"messages":[],"messages":[]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_jsonl(duplicate)

    blank = tmp_path / "blank.jsonl"
    blank.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blank JSONL records"):
        load_jsonl(blank)


def test_tool_contract_fails_closed_on_malformed_linkage() -> None:
    assert ChatMessage("tool", "", tool_call_id="call-1").content == ""
    with pytest.raises(ValueError, match="only assistant"):
        ChatMessage(
            "user",
            "question",
            tool_calls=(ToolCall("call-1", ToolFunction("lookup", '{"id":7}')),),
        )
    with pytest.raises(ValueError, match="valid strict JSON"):
        ToolFunction("lookup", "not-json")
    with pytest.raises(ValueError, match="only tool messages"):
        ChatMessage("assistant", "answer", tool_call_id="call-1")


@pytest.mark.parametrize("key", ["continue_final_message", "return_assistant_tokens_mask"])
def test_collator_rejects_chat_template_controls_that_break_flat_prefixes(key: str) -> None:
    with pytest.raises(ValueError, match="cannot override"):
        AssistantOnlyCollator(
            TinyTokenizer(),
            max_length=16,
            chat_template_kwargs={key: True},
        )
