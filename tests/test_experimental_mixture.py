import torch
from torch import nn
from queue import Queue
import sys
from types import ModuleType, SimpleNamespace

from molor.experimental import BASE_ROUTE, GenerationSettings, MixtureEngine, MixtureLinear
from molor.experimental.mixture import _trim_chat_to_budget


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs)


def tiny_engine() -> MixtureEngine:
    engine = MixtureEngine.attach(TinyModel(), base_model_id="tiny", target_modules=("q_proj",))
    engine.add_expert("left", rank=1, alpha=1.0)
    engine.add_expert("right", rank=1, alpha=1.0)
    layer = engine.layers["q_proj"]
    with torch.no_grad():
        layer.experts["left"].lora_A.copy_(torch.tensor([[1.0, 0.0]]))
        layer.experts["left"].lora_B.copy_(torch.tensor([[1.0], [0.0]]))
        layer.experts["right"].lora_A.copy_(torch.tensor([[0.0, 1.0]]))
        layer.experts["right"].lora_B.copy_(torch.tensor([[0.0], [1.0]]))
    return engine


def test_standard_forward_uses_manual_three_way_route() -> None:
    engine = tiny_engine()
    assert isinstance(engine.model.q_proj, MixtureLinear)
    engine.set_route({BASE_ROUTE: 0.2, "left": 0.3, "right": 0.5})

    output = engine.model(torch.tensor([[2.0, 4.0]]))

    torch.testing.assert_close(output, torch.tensor([[2.6, 6.0]]))
    assert not engine.model.q_proj.base.weight.requires_grad


def test_trainable_expert_and_context_route_are_isolated() -> None:
    engine = tiny_engine()
    selected = engine.set_trainable_expert("left")
    assert selected and all(parameter.requires_grad for parameter in selected)
    assert not any(parameter.requires_grad for parameter in engine.expert_parameters("right"))
    assert engine.route == {"left": 1.0}

    inputs = torch.tensor([[2.0, 4.0]])
    with engine.use_route({"right": 1.0}):
        torch.testing.assert_close(engine.model(inputs), torch.tensor([[2.0, 8.0]]))
    torch.testing.assert_close(engine.model(inputs), torch.tensor([[4.0, 4.0]]))


def test_stream_chat_reports_text_and_terminal_stats(monkeypatch) -> None:
    stop = object()

    class FakeTextIteratorStreamer:
        def __init__(self, tokenizer, *, skip_prompt, timeout, **kwargs) -> None:
            self.tokenizer = tokenizer
            self.skip_prompt = skip_prompt
            self.timeout = timeout
            self.queue = Queue()
            self.first = True

        def put(self, token_ids) -> None:
            if self.first and self.skip_prompt:
                self.first = False
                return
            self.queue.put(self.tokenizer.decode(token_ids.flatten().tolist()))

        def end(self) -> None:
            self.queue.put(stop)

        def __iter__(self):
            return self

        def __next__(self):
            value = self.queue.get(timeout=self.timeout)
            if value is stop:
                raise StopIteration
            return value

    transformers = ModuleType("transformers")
    transformers.StoppingCriteria = object
    transformers.StoppingCriteriaList = list
    transformers.TextIteratorStreamer = FakeTextIteratorStreamer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs) -> str:
            return "prompt"

        def __call__(self, prompt, *, return_tensors):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, token_ids) -> str:
            return {10: "hello ", 11: "world"}[token_ids[0]]

    class FakeCausalModel(TinyModel):
        config = SimpleNamespace(max_position_embeddings=32)

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 2)

        def get_input_embeddings(self):
            return self.embed

        def generate(self, *, input_ids, streamer, stopping_criteria, **kwargs) -> None:
            streamer.put(input_ids)
            for token_id in (10, 11):
                if stopping_criteria[0]():
                    break
                streamer.put(torch.tensor([token_id]))
            streamer.end()

    engine = MixtureEngine.attach(
        FakeCausalModel(), FakeTokenizer(), base_model_id="tiny", target_modules=("q_proj",)
    )
    updates = list(
        engine.stream_chat(
            [{"role": "user", "content": "hello"}],
            settings=GenerationSettings(max_new_tokens=4),
        )
    )

    assert "".join(update.text for update in updates) == "hello world"
    assert updates[-1].finished
    assert updates[-1].stats.prompt_tokens == 3
    assert updates[-1].stats.completion_tokens == 2
    assert updates[-1].stats.context_limit == 32
    assert updates[-1].stats.finish_reason == "stop"


def test_context_budget_drops_only_the_oldest_whole_turn() -> None:
    class WordTokenizer:
        def apply_chat_template(self, messages, **kwargs) -> str:
            return " ".join(message["content"] for message in messages)

        def __call__(self, prompt, *, return_tensors):
            count = len(prompt.split())
            return {"input_ids": torch.arange(count).unsqueeze(0)}

    prepared = _trim_chat_to_budget(
        WordTokenizer(),
        [
            {"role": "system", "content": "keep system"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ],
        max_prompt_tokens=4,
        template_kwargs={},
    )

    assert prepared.messages == (
        {"role": "system", "content": "keep system"},
        {"role": "user", "content": "new question"},
    )
    assert (prepared.prompt_tokens, prepared.dropped_turns, prepared.dropped_tokens) == (4, 1, 4)
