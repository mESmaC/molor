from dataclasses import replace

import pytest

from molor import AdapterBankManifest, RouterConfig
from molor.manifests import SAFETENSORS_MEDIA_TYPE
from molor.memory import MemoryEvent, MemoryEventKind, RetrievalHit, SQLiteMemoryStore
from molor.runtime import (
    AbstainingMemoryGate,
    BoundedRetrievalRenderer,
    CapabilityRouteDecision,
    ChatMessage,
    GenerationRequest,
    NullCapabilityRouteSelector,
    RouterBindingManifest,
    RuntimeCoordinator,
    StaticMemoryVersionResolver,
    memory_bank_manifest,
)

from test_manifests import adapter, artifact, digest
from test_memory_store import job, observation, snapshot, version


SCOPE = "user/alice"


class ByteTokenCounter:
    fingerprint = digest("8")

    def count_tokens(self, text: str) -> int:
        return len(text.encode("utf-8"))


class OneTokenCounter:
    fingerprint = digest("9")

    def count_tokens(self, text: str) -> int:
        return 1


def bank() -> AdapterBankManifest:
    return AdapterBankManifest(
        "skills",
        "1",
        (adapter("math", "1"), adapter("code", "2")),
    )


def binding(capability_bank: AdapterBankManifest) -> RouterBindingManifest:
    return RouterBindingManifest(
        bank_fingerprint=capability_bank.fingerprint,
        route_ids=capability_bank.route_ids,
        router_config=RouterConfig(
            hidden_size=16,
            num_experts=2,
            top_k=2,
        ),
        checkpoint=artifact(
            "router.safetensors",
            "4",
            media_type=SAFETENSORS_MEDIA_TYPE,
        ),
    )


def request(
    request_id: str = "request/one",
    *,
    conversation_id: str = "conversation/one",
    prompt: str = "What is Alice's favorite color?",
    memory_mode: str = "auto",
) -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        conversation_id=conversation_id,
        messages=(ChatMessage("user", prompt),),
        memory_mode=memory_mode,  # type: ignore[arg-type]
    )


def coordinator(
    store: SQLiteMemoryStore,
    *,
    resolver: object | None = None,
    renderer: object | None = None,
) -> RuntimeCoordinator:
    capability_bank = bank()
    router_binding = binding(capability_bank)
    return RuntimeCoordinator(
        capability_bank=capability_bank,
        router_binding=router_binding,
        route_selector=NullCapabilityRouteSelector(capability_bank, router_binding),
        memory_store=store,
        version_resolver=(
            StaticMemoryVersionResolver() if resolver is None else resolver  # type: ignore[arg-type]
        ),
        memory_gate=AbstainingMemoryGate(),
        retrieval_renderer=(
            BoundedRetrievalRenderer(ByteTokenCounter()) if renderer is None else renderer  # type: ignore[arg-type]
        ),
    )


def test_router_binding_covers_checkpoint_and_exact_route_order() -> None:
    capability_bank = bank()
    original = binding(capability_bank)
    changed_checkpoint = replace(
        original,
        checkpoint=artifact(
            "router.safetensors",
            "5",
            media_type=SAFETENSORS_MEDIA_TYPE,
        ),
    )

    assert original.fingerprint != changed_checkpoint.fingerprint
    original.assert_matches(capability_bank)
    with pytest.raises(ValueError, match="route order"):
        replace(original, route_ids=tuple(reversed(original.route_ids))).assert_matches(
            capability_bank
        )
    with pytest.raises(ValueError, match="safetensors"):
        replace(original, checkpoint=artifact("router.safetensors", "5"))
    mismatched_config = replace(original.router_config, num_experts=1)
    with pytest.raises(ValueError, match="expert count"):
        replace(original, router_config=mismatched_config).assert_matches(capability_bank)


def test_memory_off_skips_pinning_and_retrieval(monkeypatch) -> None:
    with SQLiteMemoryStore() as store:
        runtime = coordinator(store)

        def forbidden(*args, **kwargs):
            raise AssertionError("memory store must not be read")

        monkeypatch.setattr(store, "pin_session", forbidden)
        monkeypatch.setattr(store, "search_active_events", forbidden)
        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request(memory_mode="off"),
        )

    assert prepared.memory_route.state == "off"
    assert prepared.retrieval_records == ()
    assert prepared.retrieval_pin is None
    assert prepared.serving_pin.memory_pin is None
    assert prepared.capability_route.route_ids[-1] == "base/null"
    assert prepared.capability_route.weights[-1] == 1.0


def test_auto_memory_pins_per_request_and_retrieves_scoped_events() -> None:
    alice = observation(
        "event/alice-color",
        0,
        {"favorite_color": "blue", "person": "Alice"},
        session_id="conversation/source",
    )
    bob = observation(
        "event/bob-color",
        0,
        {"favorite_color": "green", "person": "Bob"},
        scope_id="user/bob",
        session_id="conversation/source",
    )

    with SQLiteMemoryStore() as store:
        store.append_event(alice)
        store.append_event(bob)
        runtime = coordinator(store)
        prepared = runtime.prepare(trusted_scope_id=SCOPE, request=request())

    assert prepared.memory_route.state == "unavailable"
    assert prepared.serving_pin.memory_pin is not None
    assert prepared.serving_pin.memory_pin.session_id == "request/one"
    assert tuple(record.source_event_id for record in prepared.retrieval_records) == (
        alice.event_id,
    )
    assert (
        prepared.to_dict()["retrieval_records"][0]["source_event_fingerprint"] == alice.fingerprint
    )
    assert "favorite_color" not in str(prepared.to_dict())
    cross_scope = replace(prepared.retrieval_records[0], source_scope_id="user/bob")
    with pytest.raises(ValueError, match="another memory scope"):
        replace(prepared, retrieval_records=(cross_scope,))


def test_new_request_in_one_conversation_receives_a_fresh_retrieval_cut() -> None:
    first = observation(
        "event/first",
        0,
        {"project": "molor"},
        session_id="conversation/source",
    )
    second = MemoryEvent(
        event_id="event/second",
        scope_id=SCOPE,
        session_id="conversation/source",
        sequence=1,
        kind=MemoryEventKind.OBSERVATION,
        payload={"project": "molor", "detail": "router"},
    )

    with SQLiteMemoryStore() as store:
        store.append_event(first)
        runtime = coordinator(store)
        earlier = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request("request/earlier", prompt="molor router"),
        )
        store.append_event(second)
        later = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request("request/later", prompt="molor router"),
        )

    assert tuple(record.source_event_id for record in earlier.retrieval_records) == (
        first.event_id,
    )
    assert tuple(record.source_event_id for record in later.retrieval_records) == (
        second.event_id,
        first.event_id,
    )
    assert earlier.request.conversation_id == later.request.conversation_id
    assert earlier.serving_pin.memory_pin != later.serving_pin.memory_pin


def test_promoted_memory_version_becomes_an_independent_fail_closed_bank() -> None:
    source = observation("event/source", 0, {"text": "remember this"})
    rollover = snapshot((source,))
    spec = job(rollover, cutoff=0)
    candidate = version(spec)
    with pytest.raises(ValueError, match="reserved"):
        replace(
            candidate,
            adapter=replace(candidate.adapter, adapter_id="memory/null"),
        )

    with SQLiteMemoryStore() as store:
        store.append_event(source)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        store.register_candidate(lease, candidate, now_unix_ms=400)
        store.promote_candidate(spec.job_key, now_unix_ms=500)

        runtime = coordinator(
            store,
            resolver=StaticMemoryVersionResolver({candidate.fingerprint: candidate}),
        )
        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request(prompt="remember this"),
        )

    expected_bank = memory_bank_manifest(candidate)
    assert prepared.memory_bank == expected_bank
    assert expected_bank.route_ids == (candidate.adapter.adapter_id, "memory/null")
    assert prepared.memory_route.state == "abstained"
    assert prepared.memory_route.weights == (0.0, 1.0)
    assert prepared.serving_pin.model_stack.memory_version == candidate


def test_memory_version_conflicting_with_capability_axis_falls_back_to_retrieval() -> None:
    source = observation("event/colliding-source", 0, {"text": "remember collision"})
    rollover = snapshot((source,))
    spec = job(rollover, cutoff=0)
    candidate = version(spec)
    colliding = replace(
        candidate,
        adapter=replace(candidate.adapter, adapter_id="math"),
    )

    with SQLiteMemoryStore() as store:
        store.append_event(source)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        store.register_candidate(lease, colliding, now_unix_ms=400)
        store.promote_candidate(spec.job_key, now_unix_ms=500)

        runtime = coordinator(
            store,
            resolver=StaticMemoryVersionResolver({colliding.fingerprint: colliding}),
        )
        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request(prompt="remember collision"),
        )

    assert prepared.memory_route.state == "unavailable"
    assert prepared.memory_route.attempted_version_fingerprint == colliding.fingerprint
    assert prepared.serving_pin.model_stack.memory_version is None
    assert prepared.memory_bank is None
    assert prepared.retrieval_records


def test_unverifiable_memory_version_falls_back_to_pinned_retrieval_only() -> None:
    source = observation("event/source", 0, {"text": "remember this"})
    rollover = snapshot((source,))
    spec = job(rollover, cutoff=0)
    candidate = version(spec)

    class WrongResolver:
        def resolve(self, fingerprint: str):
            return replace(candidate, version_id="memory/substituted")

    with SQLiteMemoryStore() as store:
        store.append_event(source)
        store.record_rollover(rollover, spec, now_unix_ms=0)
        lease = store.lease_next("worker/one", lease_seconds=10, now_unix_ms=100)
        assert lease is not None
        store.start_training(lease, now_unix_ms=200)
        store.start_evaluating(lease, now_unix_ms=300)
        store.register_candidate(lease, candidate, now_unix_ms=400)
        store.promote_candidate(spec.job_key, now_unix_ms=500)
        store.append_event(
            observation(
                "event/recent",
                1,
                {"text": "remember this recent detail"},
            )
        )
        runtime = coordinator(store, resolver=WrongResolver())

        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request(prompt="remember recent detail"),
        )

    assert prepared.memory_route.state == "unavailable"
    assert prepared.memory_route.attempted_version_fingerprint == candidate.fingerprint
    assert prepared.memory_route.version_fingerprint is None
    assert prepared.serving_pin.model_stack.memory_version is None
    assert prepared.serving_pin.memory_pin is None
    assert prepared.retrieval_pin is not None
    assert prepared.retrieval_pin.version_fingerprint == candidate.fingerprint
    assert prepared.retrieval_records


def test_route_selector_cannot_change_bank_identity() -> None:
    capability_bank = bank()
    router_binding = binding(capability_bank)

    class WrongSelector:
        fingerprint = digest("6")

        def select(self, supplied_request: GenerationRequest) -> CapabilityRouteDecision:
            return CapabilityRouteDecision(
                selector_fingerprint=self.fingerprint,
                selection_mode="forced",
                router_binding_fingerprint=router_binding.fingerprint,
                bank_fingerprint=digest("7"),
                route_ids=capability_bank.route_ids,
                weights=(0.0, 0.0, 1.0),
                selected_indices=(2,),
            )

    with SQLiteMemoryStore() as store:
        runtime = RuntimeCoordinator(
            capability_bank=capability_bank,
            router_binding=router_binding,
            route_selector=WrongSelector(),
            memory_store=store,
            version_resolver=StaticMemoryVersionResolver(),
            memory_gate=AbstainingMemoryGate(),
            retrieval_renderer=BoundedRetrievalRenderer(ByteTokenCounter()),
        )
        with pytest.raises(ValueError, match="different capability bank"):
            runtime.prepare(
                trusted_scope_id=SCOPE,
                request=request(memory_mode="off"),
            )


def test_request_validation_keeps_scope_out_of_browser_content() -> None:
    with pytest.raises(ValueError, match="final message"):
        GenerationRequest(
            request_id="request/invalid",
            conversation_id="conversation/one",
            messages=(ChatMessage("assistant", "not a prompt"),),
        )
    assert "scope" not in request().to_dict()

    with pytest.raises(ValueError, match="non-empty"):
        ChatMessage("user", "  \n\t")


def test_retrieval_renderer_enforces_exact_token_and_record_byte_budgets() -> None:
    first = observation(
        "event/first-budget",
        0,
        {"topic": "budget", "content": "first"},
        session_id="conversation/source",
    )
    second = MemoryEvent(
        event_id="event/second-budget",
        scope_id=SCOPE,
        session_id="conversation/source",
        sequence=1,
        kind=MemoryEventKind.OBSERVATION,
        payload={"topic": "budget", "content": "second"},
    )

    with SQLiteMemoryStore() as store:
        store.append_event(first)
        store.append_event(second)
        runtime = coordinator(
            store,
            renderer=BoundedRetrievalRenderer(
                OneTokenCounter(),
                max_tokens=1,
                max_record_bytes=1024,
            ),
        )
        prepared = runtime.prepare(
            trusted_scope_id=SCOPE,
            request=request(prompt="budget"),
        )

    assert len(prepared.retrieval_records) == 1
    assert sum(record.token_count for record in prepared.retrieval_records) == 1
    assert "untrusted-retrieved-memory" in prepared.retrieval_records[0].text
    assert "content" not in str(prepared.to_dict())

    too_small = BoundedRetrievalRenderer(
        OneTokenCounter(),
        max_tokens=10,
        max_record_bytes=32,
    )
    assert too_small.render((RetrievalHit(first, 1.0),)) == ()


def test_retrieval_renderer_detects_a_changed_tokenizer_identity() -> None:
    counter = ByteTokenCounter()
    renderer = BoundedRetrievalRenderer(counter)
    counter.fingerprint = digest("a")

    with pytest.raises(ValueError, match="identity changed"):
        renderer.render(())
