"""A vectorized bank of independent LoRA expert deltas."""

from collections.abc import Iterable, Iterator
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from molor.config import LoRABankConfig
from molor.identity import fingerprint_json, validate_sha256
from molor.manifests import AdapterBankManifest
from molor.router import RouterOutput


class LoRAExpert(nn.Module):
    """One low-rank delta for one linear projection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, **factory_kwargs))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class LoRAExpertBank(nn.Module):
    """Compute and mix complete LoRA deltas without cross-expert terms."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRABankConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        bank_manifest: AdapterBankManifest | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        if module_name is not None and (
            not isinstance(module_name, str)
            or not module_name
            or len(module_name) > 512
            or any(ord(character) < 32 for character in module_name)
        ):
            raise ValueError("module_name must be a bounded non-empty string or None")
        self.in_features = in_features
        self.out_features = out_features
        self.module_name = module_name
        self.config = config
        self._bank_fingerprint: str | None = None
        self._route_ids: tuple[str, ...] = ()
        self._bank_manifest: AdapterBankManifest | None = None
        self._parameters_verified = False
        self.dropout = nn.Dropout(config.dropout)
        self.experts = nn.ModuleList(
            LoRAExpert(
                in_features,
                out_features,
                config.rank,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_experts)
        )
        if bank_manifest is not None:
            self.bind_bank_manifest(bank_manifest)

    @property
    def bank_fingerprint(self) -> str | None:
        return self._bank_fingerprint

    @property
    def route_ids(self) -> tuple[str, ...]:
        return self._route_ids

    @property
    def parameters_verified(self) -> bool:
        return self._parameters_verified

    @property
    def verified_bank_fingerprint(self) -> str | None:
        return self._bank_fingerprint if self._parameters_verified else None

    @property
    def runtime_contract_fingerprint(self) -> str:
        return fingerprint_json(
            {
                "config": self.config.to_dict(),
                "in_features": self.in_features,
                "out_features": self.out_features,
                "dtype": str(self.experts[0].lora_A.dtype).removeprefix("torch."),
                "module_name": self.module_name,
            }
        )

    def bind_bank_manifest(self, manifest: AdapterBankManifest) -> None:
        """Bind positional expert parameters to one exact whole-adapter bank."""

        if not isinstance(manifest, AdapterBankManifest):
            raise TypeError("manifest must be an AdapterBankManifest")
        manifest.validate_lora_config(self.config)
        runtime_dtype = str(self.experts[0].lora_A.dtype).removeprefix("torch.")
        manifest.validate_runtime_module(
            module_name=self.module_name,
            in_features=self.in_features,
            out_features=self.out_features,
            dtype=runtime_dtype,
        )
        if self._bank_fingerprint is not None and self._bank_fingerprint != manifest.fingerprint:
            raise ValueError("LoRA expert bank is already bound to a different adapter bank")
        self._bank_fingerprint = manifest.fingerprint
        self._route_ids = manifest.route_ids
        self._bank_manifest = manifest
        self._parameters_verified = False

    def confirm_loaded_artifacts(
        self,
        manifest: AdapterBankManifest,
        *,
        artifact_sha256s: Mapping[str, str],
        tensor_schema_sha256: str,
    ) -> None:
        """Accept the loader's digest/schema evidence after safetensors materialization."""

        if not isinstance(manifest, AdapterBankManifest):
            raise TypeError("manifest must be an AdapterBankManifest")
        if not isinstance(artifact_sha256s, Mapping):
            raise TypeError("artifact_sha256s must be a mapping")
        self.bind_bank_manifest(manifest)
        expected_artifacts = {
            f"{adapter.adapter_id}:{artifact.logical_name}": artifact.sha256
            for adapter in manifest.adapters
            for artifact in adapter.weights
        }
        supplied_artifacts = {
            key: validate_sha256(value, f"artifact_sha256s[{key!r}]")
            for key, value in artifact_sha256s.items()
        }
        if supplied_artifacts != expected_artifacts:
            raise ValueError("verified adapter artifacts do not match the bank manifest")
        if (
            validate_sha256(tensor_schema_sha256, "tensor_schema_sha256")
            != manifest.lora.tensor_schema_sha256
        ):
            raise ValueError("loaded adapter tensor schema does not match the bank manifest")
        self._parameters_verified = True

    def get_extra_state(self) -> dict[str, object]:
        """Persist route identity alongside expert tensors in state dictionaries."""

        return {
            "schema_version": "molor.lora-bank-binding.v1",
            "runtime_contract_fingerprint": self.runtime_contract_fingerprint,
            "bank_fingerprint": self._bank_fingerprint,
            "route_ids": list(self._route_ids),
        }

    def set_extra_state(self, state: object) -> None:
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != "molor.lora-bank-binding.v1"
        ):
            raise RuntimeError("invalid or unsupported LoRA bank binding state")
        runtime_contract_fingerprint = validate_sha256(
            state.get("runtime_contract_fingerprint"),
            "runtime_contract_fingerprint",
        )
        if runtime_contract_fingerprint != self.runtime_contract_fingerprint:
            raise RuntimeError("LoRA checkpoint runtime contract differs from this bank")
        fingerprint = state.get("bank_fingerprint")
        route_ids_value = state.get("route_ids")
        if fingerprint is None:
            if route_ids_value not in ([], ()):
                raise RuntimeError("unbound LoRA bank state must not contain route IDs")
            route_ids: tuple[str, ...] = ()
        else:
            fingerprint = validate_sha256(fingerprint, "bank_fingerprint")
            if not isinstance(route_ids_value, (list, tuple)) or not route_ids_value:
                raise RuntimeError("bound LoRA bank state requires ordered route IDs")
            if any(not isinstance(route_id, str) or not route_id for route_id in route_ids_value):
                raise RuntimeError("LoRA bank binding contains invalid route IDs")
            route_ids = tuple(route_ids_value)
            if len(route_ids) != self.config.num_routes:
                raise RuntimeError("LoRA bank binding route count does not match runtime config")
        if self._bank_fingerprint is not None and fingerprint != self._bank_fingerprint:
            raise RuntimeError("LoRA checkpoint belongs to a different adapter bank")
        if self._route_ids and route_ids != self._route_ids:
            raise RuntimeError("LoRA checkpoint route order differs from the bound bank")
        self._bank_fingerprint = fingerprint
        self._route_ids = route_ids
        self._bank_manifest = None
        self._parameters_verified = False

    def forward(
        self,
        inputs: Tensor,
        routing: Tensor | RouterOutput,
        *,
        bank_fingerprint: str | None = None,
    ) -> Tensor:
        """Return the weighted adapter delta for `inputs`.

        `routing_weights` may be global (`routes`), per sequence (`batch, routes`), or
        per token (`batch, sequence, routes`). Sequence weights are broadcast over tokens.
        When configured, the last route is the exact zero/null expert.
        """

        if isinstance(routing, RouterOutput):
            if bank_fingerprint is not None:
                raise ValueError("bank_fingerprint must not be repeated for a RouterOutput")
            routing_weights = routing.weights
            supplied_fingerprint = routing.bank_fingerprint
            supplied_route_ids = routing.route_ids
        else:
            routing_weights = routing
            supplied_fingerprint = (
                None
                if bank_fingerprint is None
                else validate_sha256(bank_fingerprint, "bank_fingerprint")
            )
            supplied_route_ids = ()

        if self._bank_fingerprint is not None and not self._parameters_verified:
            raise RuntimeError(
                "bound LoRA parameters are not artifact-verified; the loader must confirm "
                "safetensors digests and tensor schema"
            )
        if supplied_fingerprint != self._bank_fingerprint:
            raise ValueError("routing weights and LoRA parameters belong to different banks")
        if supplied_route_ids and supplied_route_ids != self._route_ids:
            raise ValueError("routing output route order differs from the LoRA bank")
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {inputs.shape[-1]}")
        if routing_weights.shape[-1] != self.config.num_routes:
            raise ValueError(
                f"expected {self.config.num_routes} route weights, got {routing_weights.shape[-1]}"
            )

        lora_a = torch.stack([expert.lora_A for expert in self.experts], dim=0)
        lora_b = torch.stack([expert.lora_B for expert in self.experts], dim=0)
        dropped = self.dropout(inputs)
        low_rank = torch.einsum("...i,eri->...er", dropped, lora_a)
        expert_deltas = torch.einsum("...er,eor->...eo", low_rank, lora_b)

        weights = routing_weights[..., : self.config.num_experts]
        while weights.ndim < expert_deltas.ndim - 1:
            weights = weights.unsqueeze(-2)
        weights = weights.to(device=expert_deltas.device, dtype=expert_deltas.dtype)

        return (expert_deltas * weights.unsqueeze(-1)).sum(dim=-2) * self.config.scaling

    def expert_parameters(self, expert_index: int) -> Iterator[nn.Parameter]:
        """Return only one expert's parameters for an isolated optimizer."""

        if not 0 <= expert_index < len(self.experts):
            raise IndexError(f"expert index {expert_index} is out of range")
        return self.experts[expert_index].parameters()

    def set_trainable_experts(self, expert_indices: Iterable[int]) -> None:
        """Enable gradients only for the requested experts."""

        selected = set(expert_indices)
        invalid = selected.difference(range(len(self.experts)))
        if invalid:
            raise IndexError(f"expert indices are out of range: {sorted(invalid)}")
        for index, expert in enumerate(self.experts):
            expert.requires_grad_(index in selected)
