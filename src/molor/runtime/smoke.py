"""Local deterministic smoke runtime entry point.

This module deliberately loads no model or adapter tensors.  It composes the real
coordinator, tokenizer-bound prompt renderer, deterministic backend, generation service,
and authenticated loopback HTTP transport so their integration can be exercised before
a verified model backend is configured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import hashlib
import os
import re
import sys
from typing import IO, Iterator, Protocol

from molor.config import RouterConfig
from molor.manifests import (
    JSON_MEDIA_TYPE,
    SAFETENSORS_MEDIA_TYPE,
    AdapterBankManifest,
    AdapterManifest,
    ArtifactDescriptor,
    BaseModelManifest,
    LoRAFormatManifest,
    TargetModuleManifest,
)
from molor.memory import SQLiteMemoryStore
from molor.runtime.coordinator import (
    AbstainingMemoryGate,
    BoundedRetrievalRenderer,
    NullCapabilityRouteSelector,
    RuntimeCoordinator,
    StaticMemoryVersionResolver,
)
from molor.runtime.deterministic import (
    ByteRuntimeTokenizer,
    DeterministicGenerationBackend,
)
from molor.runtime.execution import (
    BackendDescriptor,
    DecodingConfig,
    TokenizerPromptRenderer,
)
from molor.runtime.http import (
    RunningRuntimeServer,
    RuntimeHttpConfig,
    start_runtime_server,
)
from molor.runtime.manifests import ModelStackManifest, RouterBindingManifest
from molor.runtime.service import CoordinatedGenerationService, StaticPreparationTelemetry


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_SCOPE_ID = "user/local-smoke"
SMOKE_RESPONSE = (
    "[DETERMINISTIC SMOKE MODE - NO MODEL WEIGHTS LOADED]\n\n"
    "The authenticated loopback transport, immutable coordinator, prompt renderer, "
    "cancellation path, and streaming metrics are operational. Responses in this mode "
    "are fixed test data."
)

_HELP = """molor deterministic smoke runtime (NO MODEL WEIGHTS LOADED)

This command accepts no configuration arguments. Set these environment variables:
  MOLOR_RUNTIME_TOKEN       required 32-512 character URL-safe bearer token
  MOLOR_RUNTIME_HOST        optional literal 127.0.0.1 or ::1 (default 127.0.0.1)
  MOLOR_RUNTIME_PORT        optional port (default 8765)
  MOLOR_RUNTIME_SCOPE_ID    optional fixed trusted scope (default user/local-smoke)
"""


class SmokeConfigurationError(ValueError):
    """A safe local startup configuration error."""


class SmokeServerStarter(Protocol):
    def __call__(
        self,
        config: RuntimeHttpConfig,
        service: CoordinatedGenerationService,
    ) -> RunningRuntimeServer: ...


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_artifact(
    logical_name: str,
    media_type: str,
    identity: str,
) -> ArtifactDescriptor:
    """Describe a smoke-only sentinel identity that is never materialized or loaded."""

    payload = f"molor deterministic smoke sentinel: {identity}\n".encode()
    return ArtifactDescriptor(
        logical_name=logical_name,
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def build_smoke_capability_bank() -> AdapterBankManifest:
    """Build the stable synthetic bank used only to exercise coordinator identities."""

    base = BaseModelManifest(
        model_id="molor/deterministic-smoke-no-model",
        requested_revision="deterministic-smoke-v1",
        resolved_revision=_digest("molor deterministic smoke base revision v1"),
        architecture="DeterministicSmokeNoModel",
        dtype="float32",
        quantization="not-applicable-smoke",
        config=_synthetic_artifact("smoke/base/config.json", JSON_MEDIA_TYPE, "base-config"),
        weights=(
            _synthetic_artifact(
                "smoke/base/model.safetensors",
                SAFETENSORS_MEDIA_TYPE,
                "unloaded-base-weights",
            ),
        ),
        tokenizer=(
            _synthetic_artifact(
                "smoke/tokenizer/tokenizer.json",
                JSON_MEDIA_TYPE,
                "byte-runtime-tokenizer",
            ),
        ),
        special_tokens=_synthetic_artifact(
            "smoke/tokenizer/special_tokens_map.json",
            JSON_MEDIA_TYPE,
            "no-special-tokens",
        ),
    )
    lora = LoRAFormatManifest(
        rank=1,
        alpha=1,
        dropout=0,
        target_modules=("smoke_projection",),
        tensor_schema_sha256=_digest("molor deterministic smoke lora tensor schema v1"),
        module_shapes=(TargetModuleManifest("smoke_projection", 1, 1),),
        dtype="float32",
        format_name="deterministic-smoke-sentinel",
        format_version="1",
    )
    unloaded_adapter = AdapterManifest(
        adapter_id="smoke/unloaded-expert",
        revision="deterministic-smoke-v1",
        base=base,
        lora=lora,
        weights=(
            _synthetic_artifact(
                "smoke/adapter/adapter_model.safetensors",
                SAFETENSORS_MEDIA_TYPE,
                "unloaded-adapter-weights",
            ),
        ),
        peft_config=_synthetic_artifact(
            "smoke/adapter/adapter_config.json",
            JSON_MEDIA_TYPE,
            "unloaded-adapter-config",
        ),
        training_run_sha256=_digest("molor deterministic smoke training sentinel v1"),
        tags=("deterministic-smoke", "never-loaded"),
    )
    return AdapterBankManifest(
        bank_id="smoke/capability",
        revision="deterministic-smoke-v1",
        adapters=(unloaded_adapter,),
    )


def build_smoke_router_binding(bank: AdapterBankManifest) -> RouterBindingManifest:
    if not isinstance(bank, AdapterBankManifest):
        raise TypeError("bank must be an AdapterBankManifest")
    return RouterBindingManifest(
        bank_fingerprint=bank.fingerprint,
        route_ids=bank.route_ids,
        router_config=RouterConfig(
            hidden_size=1,
            num_experts=1,
            top_k=1,
            router_hidden_size=None,
        ),
        checkpoint=_synthetic_artifact(
            "smoke/router/router.safetensors",
            SAFETENSORS_MEDIA_TYPE,
            "unloaded-router-checkpoint",
        ),
    )


def build_smoke_service() -> CoordinatedGenerationService:
    """Compose the real planning/execution path around a fixed no-model backend."""

    tokenizer = ByteRuntimeTokenizer()
    capability_bank = build_smoke_capability_bank()
    router_binding = build_smoke_router_binding(capability_bank)
    route_selector = NullCapabilityRouteSelector(capability_bank, router_binding)
    retrieval_renderer = BoundedRetrievalRenderer(tokenizer)

    @contextmanager
    def coordinator_factory() -> Iterator[RuntimeCoordinator]:
        # A fresh in-memory store preserves SQLite's creating-thread affinity.  This
        # smoke runtime intentionally has no durable memory or consolidation worker.
        with SQLiteMemoryStore() as memory_store:
            yield RuntimeCoordinator(
                capability_bank=capability_bank,
                router_binding=router_binding,
                route_selector=route_selector,
                memory_store=memory_store,
                version_resolver=StaticMemoryVersionResolver(),
                memory_gate=AbstainingMemoryGate(),
                retrieval_renderer=retrieval_renderer,
            )

    stack = ModelStackManifest(capability_bank)
    descriptor = BackendDescriptor(
        backend_id="deterministic/smoke-v1",
        model_id=stack.base.model_id,
        backend_mode="DETERMINISTIC SMOKE - NO MODEL WEIGHTS",
        quantization=stack.base.quantization,
        device="cpu/control-plane-only",
        context_limit=tokenizer.context_limit,
        tokenizer_fingerprint=tokenizer.fingerprint,
        stack_fingerprint=stack.fingerprint,
        metrics_estimated=False,
    )
    backend = DeterministicGenerationBackend(
        descriptor=descriptor,
        response_text=SMOKE_RESPONSE,
        chunk_tokens=24,
    )
    return CoordinatedGenerationService(
        coordinator_factory=coordinator_factory,
        renderer=TokenizerPromptRenderer(tokenizer),
        decoding=DecodingConfig(max_new_tokens=512),
        backend=backend,
        telemetry_provider=StaticPreparationTelemetry(
            retrieval_latency_ms=0,
            consolidation_state="deterministic-smoke/no-memory-worker",
        ),
    )


def load_smoke_http_config(
    environment: Mapping[str, str | None] | None = None,
) -> RuntimeHttpConfig:
    """Load non-secret settings and the required bearer secret from the environment."""

    selected = os.environ if environment is None else environment
    token = selected.get("MOLOR_RUNTIME_TOKEN")
    if token is None or token == "":
        raise SmokeConfigurationError("MOLOR_RUNTIME_TOKEN is required.")
    host = selected.get("MOLOR_RUNTIME_HOST") or DEFAULT_HOST
    raw_port = selected.get("MOLOR_RUNTIME_PORT") or str(DEFAULT_PORT)
    if re.fullmatch(r"[0-9]+", raw_port) is None:
        raise SmokeConfigurationError("MOLOR_RUNTIME_PORT must be an integer between 0 and 65535.")
    port = int(raw_port)
    scope_id = selected.get("MOLOR_RUNTIME_SCOPE_ID") or DEFAULT_SCOPE_ID
    try:
        return RuntimeHttpConfig(
            token=token,
            trusted_scope_id=scope_id,
            host=host,
            port=port,
        )
    except ValueError as exc:
        raise SmokeConfigurationError(str(exc)) from None


def start_smoke_server(config: RuntimeHttpConfig) -> RunningRuntimeServer:
    """Start the deterministic stack behind the authenticated HTTP transport."""

    return start_runtime_server(config, build_smoke_service())


def main(
    argv: Sequence[str] | None = None,
    *,
    environment: Mapping[str, str | None] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
    server_starter: SmokeServerStarter = start_runtime_server,
) -> int:
    """Run until interrupted; bearer secrets are never accepted as CLI arguments."""

    arguments = tuple(sys.argv[1:] if argv is None else argv)
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    if arguments in (("-h",), ("--help",)):
        print(_HELP, file=output, end="")
        return 0
    if arguments:
        print(
            "This command accepts no arguments; configure it only through the documented "
            "environment variables.",
            file=errors,
        )
        return 2

    try:
        config = load_smoke_http_config(environment)
        service = build_smoke_service()
        running = server_starter(config, service)
    except (OSError, SmokeConfigurationError, ValueError) as exc:
        print(f"Unable to start deterministic smoke runtime: {exc}", file=errors)
        return 1

    print("MOLOR DETERMINISTIC SMOKE RUNTIME - NO MODEL WEIGHTS LOADED", file=output)
    print(f"Listening on {running.endpoint}", file=output)
    print("Press Ctrl-C to stop.", file=output)
    output.flush()
    try:
        while running.thread.is_alive():
            running.thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("Stopping deterministic smoke runtime.", file=output)
    finally:
        running.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
