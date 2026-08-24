"""Immutable identities for a complete model stack and one serving request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from molor.config import RouterConfig
from molor.identity import fingerprint_json, validate_sha256
from molor.manifests import (
    SAFETENSORS_MEDIA_TYPE,
    AdapterBankManifest,
    ArtifactDescriptor,
    BaseModelManifest,
)
from molor.memory import MemoryVersionManifest, SessionMemoryPin


@dataclass(frozen=True, slots=True)
class RouterBindingManifest:
    """Exact router behavior bound to one ordered capability bank.

    A bank fingerprint alone identifies the available actions, but not the router
    checkpoint or the routing semantics that select among them.  This manifest is
    therefore part of every prepared serving request.
    """

    schema_version: ClassVar[str] = "molor.router-artifact-binding.v1"

    bank_fingerprint: str
    route_ids: tuple[str, ...]
    router_config: RouterConfig
    checkpoint: ArtifactDescriptor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "bank_fingerprint",
            validate_sha256(self.bank_fingerprint, "bank_fingerprint"),
        )
        route_ids = tuple(self.route_ids)
        if not route_ids:
            raise ValueError("route_ids must not be empty")
        if any(not isinstance(route_id, str) or not route_id for route_id in route_ids):
            raise ValueError("route_ids must contain non-empty strings")
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("route_ids must not contain duplicates")
        if not isinstance(self.router_config, RouterConfig):
            raise TypeError("router_config must be a RouterConfig")
        if not isinstance(self.checkpoint, ArtifactDescriptor):
            raise TypeError("checkpoint must be an ArtifactDescriptor")
        if self.checkpoint.media_type != SAFETENSORS_MEDIA_TYPE:
            raise ValueError("router checkpoint must use the safetensors media type")
        object.__setattr__(self, "route_ids", route_ids)

    def assert_matches(self, bank: AdapterBankManifest) -> None:
        """Reject a router checkpoint attached to another action space."""

        if not isinstance(bank, AdapterBankManifest):
            raise TypeError("bank must be an AdapterBankManifest")
        if self.bank_fingerprint != bank.fingerprint:
            raise ValueError("router binding belongs to a different adapter bank")
        if self.route_ids != bank.route_ids:
            raise ValueError("router binding route order differs from the adapter bank")
        bank.validate_router_config(self.router_config)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "bank_fingerprint": self.bank_fingerprint,
            "route_ids": list(self.route_ids),
            "router_config_fingerprint": self.router_config.fingerprint,
            "router_config": self.router_config.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


def memory_bank_manifest(version: MemoryVersionManifest) -> AdapterBankManifest:
    """Expose one immutable memory version as its own adapter-plus-null bank."""

    if not isinstance(version, MemoryVersionManifest):
        raise TypeError("version must be a MemoryVersionManifest")
    return AdapterBankManifest(
        bank_id=f"memory/{version.fingerprint[:24]}",
        revision=version.version_id,
        adapters=(version.adapter,),
        include_null=True,
        null_route_id="memory/null",
    )


@dataclass(frozen=True, slots=True)
class ModelStackManifest:
    """One routed bank plus the optional memory version available to its hard gate.

    The bank remains the v1 adapter/action identity. This wrapper adds the effective
    memory-base identity without altering the bank manifest or its fingerprint.
    """

    schema_version: ClassVar[str] = "molor.model-stack.v1"

    bank: AdapterBankManifest
    memory_version: MemoryVersionManifest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.bank, AdapterBankManifest):
            raise TypeError("bank must be an AdapterBankManifest")
        if self.memory_version is not None:
            if not isinstance(self.memory_version, MemoryVersionManifest):
                raise TypeError("memory_version must be a MemoryVersionManifest or None")
            if self.memory_version.base_fingerprint != self.bank.base.fingerprint:
                raise ValueError("memory version and adapter bank target different frozen bases")
            memory_adapter = self.memory_version.adapter
            if any(
                adapter.fingerprint == memory_adapter.fingerprint for adapter in self.bank.adapters
            ):
                raise ValueError("the same adapter must not occupy capability and memory axes")
            if memory_adapter.adapter_id in {adapter.adapter_id for adapter in self.bank.adapters}:
                raise ValueError("capability and memory adapters must use disjoint logical IDs")

    @property
    def base(self) -> BaseModelManifest:
        return self.bank.base

    @property
    def base_fingerprint(self) -> str:
        return self.base.fingerprint

    @property
    def bank_fingerprint(self) -> str:
        return self.bank.fingerprint

    @property
    def memory_version_fingerprint(self) -> str | None:
        if self.memory_version is None:
            return None
        return self.memory_version.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_fingerprint": self.base_fingerprint,
            "bank_fingerprint": self.bank_fingerprint,
            "bank_manifest": self.bank.to_dict(),
            "memory_version_fingerprint": self.memory_version_fingerprint,
            "memory_version_manifest": (
                None if self.memory_version is None else self.memory_version.to_dict()
            ),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ServingPinManifest:
    """One request's immutable model stack and memory-head/retrieval view."""

    schema_version: ClassVar[str] = "molor.serving-pin.v1"

    model_stack: ModelStackManifest
    memory_pin: SessionMemoryPin | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_stack, ModelStackManifest):
            raise TypeError("model_stack must be a ModelStackManifest")
        if self.memory_pin is not None and not isinstance(self.memory_pin, SessionMemoryPin):
            raise TypeError("memory_pin must be a SessionMemoryPin or None")

        version = self.model_stack.memory_version
        pin = self.memory_pin
        if version is None:
            if pin is not None and pin.version_fingerprint is not None:
                raise ValueError("memory pin references a version absent from the model stack")
            return
        if pin is None:
            raise ValueError("a memory-backed model stack requires a session memory pin")
        if pin.scope_id != version.scope_id:
            raise ValueError("memory pin and model stack belong to different scopes")
        if pin.version_fingerprint != version.fingerprint:
            raise ValueError("memory pin references a different memory version")
        if pin.consolidated_through_sequence != version.consolidated_through_sequence:
            raise ValueError("memory pin and version use different consolidation cutoffs")

    @property
    def model_stack_fingerprint(self) -> str:
        return self.model_stack.fingerprint

    @property
    def memory_pin_fingerprint(self) -> str | None:
        if self.memory_pin is None:
            return None
        return self.memory_pin.fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_stack_fingerprint": self.model_stack_fingerprint,
            "model_stack_manifest": self.model_stack.to_dict(),
            "memory_pin_fingerprint": self.memory_pin_fingerprint,
            "memory_pin": None if self.memory_pin is None else self.memory_pin.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())
