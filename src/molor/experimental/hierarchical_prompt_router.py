"""Checkpoint-backed request-time runtime for the hierarchical expert router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from molor.experimental.autorouter import (
    BASE_ROUTE,
    NULL_ROUTE_ID,
    AutoRouteDecision,
    AutoRouterPolicy,
    manual_route_decision,
)
from molor.experimental.expert_taxonomy import ExpertTaxonomy, load_expert_taxonomy
from molor.experimental.hierarchical_autorouter import (
    HierarchicalRouterPolicy,
    decide_hierarchical_route,
)
from molor.training.hierarchical_router import (
    HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA,
    HierarchicalRouterCheckpoint,
    load_hierarchical_router_checkpoint,
)
from molor.training.router import RouterExample, extract_prompt_features


def _expert_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("available_experts must be a sequence")
    names = tuple(values)
    if any(not isinstance(name, str) or not name or name == BASE_ROUTE for name in names):
        raise ValueError("available experts must be non-empty and cannot use the base route name")
    if len(names) != len(set(names)):
        raise ValueError("available experts must not contain duplicates")
    return names


def _embedding_weight(input_embeddings: nn.Module) -> Tensor:
    weight = getattr(input_embeddings, "weight", None)
    if not isinstance(weight, Tensor) or weight.ndim != 2 or not weight.is_floating_point():
        raise TypeError("input_embeddings must expose a rank-two floating tensor weight")
    return weight


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"router_config.json contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"router_config.json contains unsupported constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("router_config.json must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("router_config.json must contain an object")
    return value


def _pinned_taxonomy_path(checkpoint_directory: Path) -> Path:
    config_path = checkpoint_directory / "router_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"router config does not exist: {config_path}")
    payload = _strict_json_object(config_path)
    taxonomy = payload.get("taxonomy")
    artifact = taxonomy.get("artifact") if isinstance(taxonomy, Mapping) else None
    raw_path = artifact.get("path") if isinstance(artifact, Mapping) else None
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("hierarchical checkpoint has no pinned taxonomy artifact path")
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (checkpoint_directory / path).resolve()


class HierarchicalPromptAutoRouter:
    """A sealed hierarchy bound to the exact frozen embedding and resident adapter rack."""

    checkpoint_schema = HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA

    def __init__(
        self,
        *,
        checkpoint: HierarchicalRouterCheckpoint,
        taxonomy: ExpertTaxonomy,
        tokenizer: Any,
        input_embeddings: nn.Module,
        available_experts: Sequence[str],
    ) -> None:
        if not isinstance(checkpoint, HierarchicalRouterCheckpoint):
            raise TypeError("checkpoint must be a HierarchicalRouterCheckpoint")
        if not isinstance(taxonomy, ExpertTaxonomy):
            raise TypeError("taxonomy must be an ExpertTaxonomy")
        names = _expert_names(available_experts)
        expected = set(checkpoint.adapter_ids)
        loaded = set(names)
        if loaded != expected:
            missing = sorted(expected - loaded)
            unknown = sorted(loaded - expected)
            raise ValueError(
                "resident adapters must exactly match every hierarchical leaf "
                f"(missing={missing}, unknown={unknown})"
            )
        if checkpoint.adapter_ids != taxonomy.leaf_ids:
            raise ValueError("checkpoint adapter axis differs from the loaded taxonomy")
        weight = _embedding_weight(input_embeddings)
        model_contract = checkpoint.sidecar.get("model")
        embedding_contract = (
            model_contract.get("embedding") if isinstance(model_contract, Mapping) else None
        )
        pinned_shape = (
            embedding_contract.get("shape") if isinstance(embedding_contract, Mapping) else None
        )
        if (
            not isinstance(pinned_shape, list)
            or len(pinned_shape) != 2
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in pinned_shape
            )
        ):
            raise ValueError("checkpoint has no valid input-embedding shape contract")
        if tuple(weight.shape) != tuple(pinned_shape):
            raise ValueError("live input-embedding shape differs from the sealed model artifact")
        if weight.shape[1] != checkpoint.router.hidden_size:
            raise ValueError(
                "live input-embedding width differs from the hierarchical router checkpoint"
            )
        if getattr(tokenizer, "truncation_side", "right") != "right":
            raise ValueError("router tokenizer truncation_side must remain right")
        router_device = next(checkpoint.router.parameters()).device
        if router_device != weight.device:
            raise ValueError(
                f"router heads are on {router_device}, but input embeddings are on {weight.device}"
            )
        self.checkpoint = checkpoint
        self.taxonomy = taxonomy
        self.tokenizer = tokenizer
        self.input_embeddings = input_embeddings
        self.available_experts = names
        self.checkpoint_directory = checkpoint.directory
        self.feature_contract = checkpoint.feature_contract
        self.max_prompt_tokens = checkpoint.feature_contract.max_tokens
        self.add_special_tokens = checkpoint.feature_contract.add_special_tokens
        self.adapter_ids = checkpoint.adapter_ids
        self._available_set = frozenset(names)
        self._embedding_version = weight._version

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_directory: str | Path,
        *,
        tokenizer: Any,
        input_embeddings: nn.Module,
        available_experts: Sequence[str],
        adapters: Mapping[str, str | Path],
        expected_model_id: str | Path,
        taxonomy_path: str | Path | None = None,
    ) -> HierarchicalPromptAutoRouter:
        """Verify and load a hierarchy plus all taxonomy-bound adapter artifacts."""

        if not isinstance(adapters, Mapping):
            raise TypeError("adapters must map every resident leaf to its directory")
        directory = Path(checkpoint_directory).expanduser().resolve()
        source = (
            _pinned_taxonomy_path(directory)
            if taxonomy_path is None
            else Path(taxonomy_path).expanduser().resolve()
        )
        weight = _embedding_weight(input_embeddings)
        checkpoint = load_hierarchical_router_checkpoint(
            directory,
            taxonomy_path=source,
            adapters=adapters,
            expected_model_id=expected_model_id,
            verify_adapter_artifacts=True,
            verify_model_artifacts=True,
            device=weight.device,
        )
        taxonomy = load_expert_taxonomy(source)
        return cls(
            checkpoint=checkpoint,
            taxonomy=taxonomy,
            tokenizer=tokenizer,
            input_embeddings=input_embeddings,
            available_experts=available_experts,
        )

    def _prompt_features(self, prompt: str) -> Tensor:
        weight = _embedding_weight(self.input_embeddings)
        if weight._version != self._embedding_version:
            raise RuntimeError("frozen input embeddings changed after router initialization")
        features, _labels = extract_prompt_features(
            [RouterExample(prompt=prompt, route_id=BASE_ROUTE)],
            tokenizer=self.tokenizer,
            embedding=weight,
            route_ids=(BASE_ROUTE,),
            contract=self.feature_contract,
            batch_size=1,
        )
        if features.shape != (1, self.checkpoint.router.hidden_size):
            raise ValueError("input_embeddings returned an invalid prompt feature shape")
        return features.to(device=weight.device)

    def decide(self, prompt: str, policy: AutoRouterPolicy) -> AutoRouteDecision:
        """Route one prompt and adapt the hierarchy to the stable desktop decision contract."""

        if not isinstance(policy, AutoRouterPolicy):
            raise TypeError("policy must be an AutoRouterPolicy")
        if policy.mode == "manual":
            return manual_route_decision(self.available_experts, policy.manual_route)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("auto routing requires a non-empty current-user prompt")
        if policy.max_prompt_tokens != self.max_prompt_tokens:
            raise ValueError(
                "router max_prompt_tokens is pinned by its training contract at "
                f"{self.max_prompt_tokens}, not {policy.max_prompt_tokens}"
            )
        if policy.top_k > 2:
            raise ValueError("the hierarchical router supports at most two active adapters")
        if policy.enabled_experts:
            unknown = set(policy.enabled_experts).difference(self._available_set)
            if unknown:
                raise KeyError(f"enabled experts are not loaded: {sorted(unknown)!r}")
            enabled = policy.enabled_experts
        else:
            enabled = self.available_experts

        features = self._prompt_features(prompt)
        root_logits, conditional_logits = self.checkpoint.router.inference_logits(features[0])
        hierarchical = decide_hierarchical_route(
            self.taxonomy,
            root_logits=root_logits,
            conditional_leaf_logits=conditional_logits,
            policy=HierarchicalRouterPolicy(
                top_k=policy.top_k,
                root_temperature=policy.temperature,
                leaf_temperature=policy.temperature,
                base_bias=policy.null_bias,
                minimum_joint_probability=policy.confidence_threshold,
                enabled_leaves=tuple(enabled),
            ),
        )
        joint = dict(hierarchical.joint_probabilities)
        probability_map = {
            BASE_ROUTE: hierarchical.root_probability_map[NULL_ROUTE_ID],
            **{leaf_id: joint.get(leaf_id, 0.0) for leaf_id in self.available_experts},
        }
        probabilities = tuple(
            (route_id, probability_map[route_id])
            for route_id in (BASE_ROUTE, *self.available_experts)
        )
        applied = tuple(
            (BASE_ROUTE if route_id == NULL_ROUTE_ID else route_id, weight)
            for route_id, weight in hierarchical.applied_route
        )
        return AutoRouteDecision(
            mode="auto",
            applied_route=applied,
            probabilities=probabilities,
            selected_routes=tuple(route_id for route_id, _weight in applied),
            confidence=hierarchical.confidence,
            reason=hierarchical.reason,
            checkpoint_loaded=True,
        )


__all__ = ["HierarchicalPromptAutoRouter"]
