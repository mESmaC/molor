"""Sparse prompt- and token-routing primitives."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from molor.config import RouterConfig
from molor.identity import validate_sha256
from molor.manifests import AdapterBankManifest


@dataclass(slots=True)
class RouterOutput:
    """Router values needed for execution, training, and observability."""

    logits: Tensor
    probabilities: Tensor
    weights: Tensor
    indices: Tensor
    load_balance_loss: Tensor
    z_loss: Tensor
    bank_fingerprint: str | None = None
    route_ids: tuple[str, ...] = ()


class TopKRouter(nn.Module):
    """Route a sequence once or route each token independently."""

    def __init__(
        self,
        config: RouterConfig,
        *,
        bank_manifest: AdapterBankManifest | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._bank_fingerprint: str | None = None
        self._route_ids: tuple[str, ...] = ()
        if config.router_hidden_size is None:
            self.network = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.num_routes),
            )
        else:
            self.network = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.router_hidden_size),
                nn.SiLU(),
                nn.Linear(config.router_hidden_size, config.num_routes),
            )
        if bank_manifest is not None:
            self.bind_bank_manifest(bank_manifest)

    @property
    def bank_fingerprint(self) -> str | None:
        return self._bank_fingerprint

    @property
    def route_ids(self) -> tuple[str, ...]:
        return self._route_ids

    def bind_bank_manifest(self, manifest: AdapterBankManifest) -> None:
        """Bind checkpoint semantics to one exact ordered adapter bank."""

        if not isinstance(manifest, AdapterBankManifest):
            raise TypeError("manifest must be an AdapterBankManifest")
        manifest.validate_router_config(self.config)
        if self._bank_fingerprint is not None and self._bank_fingerprint != manifest.fingerprint:
            raise ValueError("router is already bound to a different adapter bank")
        self._bank_fingerprint = manifest.fingerprint
        self._route_ids = manifest.route_ids

    def get_extra_state(self) -> dict[str, object]:
        """Persist bank identity inside router state dictionaries."""

        return {
            "schema_version": "molor.router-binding.v1",
            "runtime_contract_fingerprint": self.config.fingerprint,
            "bank_fingerprint": self._bank_fingerprint,
            "route_ids": list(self._route_ids),
        }

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict) or state.get("schema_version") != "molor.router-binding.v1":
            raise RuntimeError("invalid or unsupported router binding state")
        runtime_contract_fingerprint = validate_sha256(
            state.get("runtime_contract_fingerprint"),
            "runtime_contract_fingerprint",
        )
        if runtime_contract_fingerprint != self.config.fingerprint:
            raise RuntimeError("router checkpoint runtime contract differs from this router")
        fingerprint = state.get("bank_fingerprint")
        route_ids_value = state.get("route_ids")
        if fingerprint is None:
            if route_ids_value not in ([], ()):
                raise RuntimeError("unbound router state must not contain route IDs")
            route_ids: tuple[str, ...] = ()
        else:
            fingerprint = validate_sha256(fingerprint, "bank_fingerprint")
            if not isinstance(route_ids_value, (list, tuple)) or not route_ids_value:
                raise RuntimeError("bound router state requires ordered route IDs")
            if any(not isinstance(route_id, str) or not route_id for route_id in route_ids_value):
                raise RuntimeError("router binding contains invalid route IDs")
            route_ids = tuple(route_ids_value)
            if len(route_ids) != self.config.num_routes:
                raise RuntimeError("router binding route count does not match router config")
        if self._bank_fingerprint is not None and fingerprint != self._bank_fingerprint:
            raise RuntimeError("router checkpoint belongs to a different adapter bank")
        if self._route_ids and route_ids != self._route_ids:
            raise RuntimeError("router checkpoint route order differs from the bound bank")
        self._bank_fingerprint = fingerprint
        self._route_ids = route_ids

    def _router_features(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must include a batch and hidden dimension")
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"expected hidden width {self.config.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if self.config.mode == "token":
            return hidden_states

        if hidden_states.ndim == 2:
            if attention_mask is not None:
                raise ValueError("attention_mask is only valid for sequence-shaped hidden states")
            return hidden_states
        if hidden_states.ndim != 3:
            raise ValueError("sequence routing expects [batch, sequence, hidden] states")

        if attention_mask is None:
            return hidden_states.mean(dim=1)
        if attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the batch and sequence dimensions")

        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
        token_count = mask.sum(dim=1, keepdim=True)
        if torch.any(token_count == 0):
            raise ValueError("every sequence must contain at least one unmasked token")
        return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / token_count

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
    ) -> RouterOutput:
        features = self._router_features(hidden_states, attention_mask)
        logits = self.network(features) / self.config.temperature
        probabilities = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)

        top_values, indices = probabilities.topk(self.config.top_k, dim=-1)
        assignment = torch.zeros_like(probabilities).scatter(-1, indices, 1.0)
        if self.config.straight_through:
            weights = assignment - probabilities.detach() + probabilities
        elif self.config.uniform_top_k:
            # This execution path is intentionally non-differentiable. Train its discrete
            # selections with a policy-gradient objective built from `probabilities`.
            weights = assignment / self.config.top_k
        else:
            if self.config.normalize_top_k:
                top_values = top_values / top_values.sum(dim=-1, keepdim=True)
            weights = torch.zeros_like(probabilities).scatter(-1, indices, top_values)

        reduction_dims = tuple(range(probabilities.ndim - 1))
        route_fraction = assignment.float().mean(dim=reduction_dims) / self.config.top_k
        mean_probability = probabilities.float().mean(dim=reduction_dims)
        load_balance_loss = self.config.num_routes * torch.sum(
            route_fraction.detach() * mean_probability
        )
        z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()

        return RouterOutput(
            logits=logits,
            probabilities=probabilities,
            weights=weights,
            indices=indices,
            load_balance_loss=load_balance_loss,
            z_loss=z_loss,
            bank_fingerprint=self._bank_fingerprint,
            route_ids=self._route_ids,
        )
