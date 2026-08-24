from __future__ import annotations

from pathlib import Path

import pytest

from molor.loading import (
    BaseLoadPolicy,
    TransformersBaseLoader,
    TransformersLoadError,
    TransformersRuntimeTokenizer,
)
from molor.runtime.execution import PromptMessage, RuntimeTokenizer

from test_loading_transformers import FakeBackend, FakeTokenizer, staged_manifest


class TemplateTokenizer(FakeTokenizer):
    def __init__(self, result: object = None) -> None:
        super().__init__()
        self.result = [3, 5, 8] if result is None else result
        self.chat_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> object:
        self.chat_calls.append((messages, kwargs))
        return self.result


def load_tokenizer(
    tmp_path: Path,
    tokenizer: object,
    *,
    context_limit: int = 2048,
):
    manifest, staged = staged_manifest(tmp_path)
    runtime = TransformersBaseLoader(FakeBackend(tokenizer=tokenizer)).load(
        manifest,
        staged,
        policy=BaseLoadPolicy(context_window_cap=context_limit),
    )
    return runtime


def requires_runtime_tokenizer(tokenizer: RuntimeTokenizer) -> RuntimeTokenizer:
    """Static structural check exercised by type checkers and runtime tests alike."""

    return tokenizer


def test_loaded_tokenizer_satisfies_runtime_prompt_contract(tmp_path: Path) -> None:
    tokenizer = TemplateTokenizer()
    runtime = load_tokenizer(tmp_path, tokenizer, context_limit=1024)
    try:
        execution_tokenizer = requires_runtime_tokenizer(runtime.runtime_tokenizer)
        messages = (
            PromptMessage("system", "System policy"),
            PromptMessage("user", "Current request"),
        )

        assert isinstance(execution_tokenizer, TransformersRuntimeTokenizer)
        assert runtime.runtime_tokenizer is runtime.token_counter
        assert execution_tokenizer.context_limit == 1024
        assert execution_tokenizer.encode_chat(
            messages,
            add_generation_prompt=True,
        ) == (3, 5, 8)
        assert tokenizer.chat_calls == [
            (
                [
                    {"role": "system", "content": "System policy"},
                    {"role": "user", "content": "Current request"},
                ],
                {
                    "tokenize": True,
                    "add_generation_prompt": True,
                    "truncation": False,
                    "padding": False,
                    "return_tensors": None,
                    "return_dict": False,
                },
            )
        ]
    finally:
        runtime.close()


def test_generation_prompt_flag_is_forwarded_exactly(tmp_path: Path) -> None:
    tokenizer = TemplateTokenizer(result=(1, 2))
    runtime = load_tokenizer(tmp_path, tokenizer)
    try:
        ids = runtime.runtime_tokenizer.encode_chat(
            (PromptMessage("user", "No generation marker"),),
            add_generation_prompt=False,
        )
        assert ids == (1, 2)
        assert tokenizer.chat_calls[0][1]["add_generation_prompt"] is False
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ([], "returned no token IDs"),
        ([[1, 2]], "invalid or non-flat"),
        ([True], "invalid or non-flat"),
        ([-1], "invalid or non-flat"),
        ([1.0], "invalid or non-flat"),
        ([32], "invalid or non-flat"),
        ({"input_ids": [1]}, "flat token list"),
        ((value for value in [1, 2]), "flat token list"),
    ],
)
def test_chat_encoding_rejects_non_exact_flat_integer_ids(
    tmp_path: Path,
    result: object,
    message: str,
) -> None:
    runtime = load_tokenizer(tmp_path, TemplateTokenizer(result=result))
    try:
        with pytest.raises(TransformersLoadError, match=message):
            runtime.runtime_tokenizer.encode_chat(
                (PromptMessage("user", "Hello"),),
                add_generation_prompt=True,
            )
    finally:
        runtime.close()


def test_chat_encoding_strictly_validates_inputs(tmp_path: Path) -> None:
    runtime = load_tokenizer(tmp_path, TemplateTokenizer())
    try:
        with pytest.raises(TypeError, match="must be boolean"):
            runtime.runtime_tokenizer.encode_chat(
                (PromptMessage("user", "Hello"),),
                add_generation_prompt=1,  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="must be a tuple"):
            runtime.runtime_tokenizer.encode_chat(
                [PromptMessage("user", "Hello")],  # type: ignore[arg-type]
                add_generation_prompt=True,
            )
        with pytest.raises(ValueError, match="must not be empty"):
            runtime.runtime_tokenizer.encode_chat((), add_generation_prompt=True)
        with pytest.raises(TypeError, match="only PromptMessage"):
            runtime.runtime_tokenizer.encode_chat(
                ({"role": "user", "content": "Hello"},),  # type: ignore[arg-type]
                add_generation_prompt=True,
            )
    finally:
        runtime.close()


def test_chat_encoding_never_falls_back_to_plain_encode(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    runtime = load_tokenizer(tmp_path, tokenizer)
    try:
        with pytest.raises(TransformersLoadError, match="apply_chat_template"):
            runtime.runtime_tokenizer.encode_chat(
                (PromptMessage("user", "Hello"),),
                add_generation_prompt=True,
            )
        assert tokenizer.encode_calls == []
    finally:
        runtime.close()


def test_chat_template_identity_cannot_drift_after_loading(tmp_path: Path) -> None:
    tokenizer = TemplateTokenizer()
    runtime = load_tokenizer(tmp_path, tokenizer)
    try:
        tokenizer.chat_template = "mutated after verification"
        with pytest.raises(TransformersLoadError, match="stable verified chat template"):
            runtime.runtime_tokenizer.encode_chat(
                (PromptMessage("user", "Hello"),),
                add_generation_prompt=True,
            )
        assert tokenizer.chat_calls == []
    finally:
        runtime.close()


def test_plain_count_remains_exact_and_allows_empty_text(tmp_path: Path) -> None:
    tokenizer = TemplateTokenizer()
    runtime = load_tokenizer(tmp_path, tokenizer)
    try:
        assert runtime.runtime_tokenizer.count_tokens("") == 0
        assert runtime.runtime_tokenizer.count_tokens("one two") == 2
        assert tokenizer.encode_calls == [("", False, False), ("one two", False, False)]
    finally:
        runtime.close()
