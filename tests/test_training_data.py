from __future__ import annotations

from molor.training import AssistantOnlyCollator, ChatExample, ChatMessage


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


def test_collator_masks_prompt_and_padding_but_learns_assistant() -> None:
    examples = [
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "short"))),
        ChatExample((ChatMessage("user", "Question"), ChatMessage("assistant", "longer"))),
    ]

    batch = AssistantOnlyCollator(TinyTokenizer(), max_length=16)(examples)

    assert batch["input_ids"].tolist() == [[1, 2, 3, 5, 9], [1, 2, 3, 6, 9]]
    assert batch["labels"].tolist() == [[-100, -100, -100, 5, 9], [-100, -100, -100, 6, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
