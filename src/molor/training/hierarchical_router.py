"""Trainable family-to-leaf prompt routing over frozen embedding features.

The hierarchy is structural rather than a post-hoc relabeling of one flat head: a root head
chooses ``base/null`` or an expert family, and one conditional head per family chooses a leaf.
The resulting leaf posterior is exactly ``P(family) * P(leaf | family)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import unicodedata

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from molor.experimental.expert_taxonomy import ExpertTaxonomy, load_expert_taxonomy
from molor.experimental.hierarchical_autorouter import BASE_ROUTE
from molor.identity import fingerprint_json, validate_sha256
from molor.training.router import (
    RouterExample,
    RouterFeatureContract,
    RouterTrainingConfig,
    cached_prompt_features,
    load_input_embedding,
    load_router_jsonl,
    load_router_tokenizer,
)


HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA = "molor.hierarchical-router-checkpoint.v1"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class HierarchicalRouterAxes:
    """Canonical route ordering shared by training, checkpoints, and inference."""

    root_routes: tuple[str, ...]
    family_leaves: tuple[tuple[str, tuple[str, ...]], ...]
    joint_routes: tuple[str, ...]

    @classmethod
    def from_taxonomy(cls, taxonomy: ExpertTaxonomy) -> HierarchicalRouterAxes:
        if not isinstance(taxonomy, ExpertTaxonomy):
            raise TypeError("taxonomy must be an ExpertTaxonomy")
        root_routes = (BASE_ROUTE, *(family.family_id for family in taxonomy.families))
        family_leaves = tuple(
            (family.family_id, tuple(leaf.leaf_id for leaf in family.leaves))
            for family in taxonomy.families
        )
        return cls(
            root_routes=root_routes,
            family_leaves=family_leaves,
            joint_routes=(BASE_ROUTE, *taxonomy.leaf_ids),
        )

    @property
    def leaf_to_family(self) -> Mapping[str, str]:
        return {
            leaf_id: family_id for family_id, leaf_ids in self.family_leaves for leaf_id in leaf_ids
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_routes": list(self.root_routes),
            "family_leaves": {
                family_id: list(leaf_ids) for family_id, leaf_ids in self.family_leaves
            },
            "joint_routes": list(self.joint_routes),
        }


@dataclass(slots=True)
class HierarchicalRouterOutput:
    root_logits: Tensor
    conditional_leaf_logits: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class HierarchicalRouterTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    metrics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class HierarchicalRouterCheckpoint:
    """Integrity-verified hierarchy ready to supply logits to request-time routing."""

    directory: Path
    router: HierarchicalRouterHeads
    axes: HierarchicalRouterAxes
    feature_contract: RouterFeatureContract
    model_id: str
    adapter_ids: tuple[str, ...]
    sidecar: Mapping[str, Any]


def _head(hidden_size: int, output_size: int, bottleneck: int | None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.LayerNorm(hidden_size)]
    if bottleneck is None:
        layers.append(nn.Linear(hidden_size, output_size))
    else:
        layers.extend(
            (
                nn.Linear(hidden_size, bottleneck),
                nn.SiLU(),
                nn.Linear(bottleneck, output_size),
            )
        )
    return nn.Sequential(*layers)


class HierarchicalRouterHeads(nn.Module):
    """One root head plus an independent conditional head for each expert family."""

    def __init__(
        self,
        *,
        hidden_size: int,
        axes: HierarchicalRouterAxes,
        router_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if router_hidden_size is not None and (
            not isinstance(router_hidden_size, int)
            or isinstance(router_hidden_size, bool)
            or router_hidden_size < 1
        ):
            raise ValueError("router_hidden_size must be a positive integer or None")
        if len(axes.root_routes) < 2 or axes.root_routes[0] != BASE_ROUTE:
            raise ValueError("root routes must begin with base/null and include a family")
        if not axes.family_leaves:
            raise ValueError("the hierarchy must contain at least one family")
        if tuple(family for family, _leaves in axes.family_leaves) != axes.root_routes[1:]:
            raise ValueError("family head order must exactly match the root family order")
        if any(not leaves for _family, leaves in axes.family_leaves):
            raise ValueError("every family must contain at least one leaf")
        expected_joint = (
            BASE_ROUTE,
            *(leaf for _family, leaves in axes.family_leaves for leaf in leaves),
        )
        if axes.joint_routes != expected_joint or len(set(expected_joint)) != len(expected_joint):
            raise ValueError("joint routes must be unique and follow canonical family order")

        self.hidden_size = hidden_size
        self.axes = axes
        self.router_hidden_size = router_hidden_size
        self.root_network = _head(hidden_size, len(axes.root_routes), router_hidden_size)
        self.leaf_networks = nn.ModuleDict(
            {
                family_id: _head(hidden_size, len(leaves), router_hidden_size)
                for family_id, leaves in axes.family_leaves
            }
        )

    def forward(self, features: Tensor) -> HierarchicalRouterOutput:
        if features.ndim != 2 or features.shape[1] != self.hidden_size:
            raise ValueError(f"features must have shape [rows, {self.hidden_size}]")
        return HierarchicalRouterOutput(
            root_logits=self.root_network(features),
            conditional_leaf_logits={
                family_id: self.leaf_networks[family_id](features)
                for family_id, _leaves in self.axes.family_leaves
            },
        )

    def _joint_probabilities_from_output(self, output: HierarchicalRouterOutput) -> Tensor:
        root = torch.softmax(output.root_logits.float(), dim=-1)
        columns = [root[:, :1]]
        for root_index, (family_id, _leaves) in enumerate(self.axes.family_leaves, start=1):
            conditional = torch.softmax(output.conditional_leaf_logits[family_id].float(), dim=-1)
            columns.append(root[:, root_index : root_index + 1] * conditional)
        probabilities = torch.cat(columns, dim=-1)
        if probabilities.shape[1] != len(self.axes.joint_routes):
            raise RuntimeError("hierarchical probability axis does not match its route contract")
        return probabilities

    def joint_probabilities(self, features: Tensor) -> Tensor:
        """Return normalized ``base + all leaves`` probabilities in ``joint_routes`` order."""

        return self._joint_probabilities_from_output(self(features))

    def inference_logits(
        self, features: Tensor
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Return one row of raw logits keyed for ``decide_hierarchical_route``."""

        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape != (1, self.hidden_size):
            raise ValueError(f"features must describe one row with width {self.hidden_size}")
        parameter = next(self.parameters())
        features = features.to(device=parameter.device, dtype=parameter.dtype)
        with torch.inference_mode():
            output = self(features)
        root = {
            route_id: float(output.root_logits[0, index].float().item())
            for index, route_id in enumerate(self.axes.root_routes)
        }
        conditional = {
            family_id: {
                leaf_id: float(output.conditional_leaf_logits[family_id][0, index].float().item())
                for index, leaf_id in enumerate(leaves)
            }
            for family_id, leaves in self.axes.family_leaves
        }
        return root, conditional


@dataclass(frozen=True, slots=True)
class _EncodedTargets:
    flat: Tensor
    root: Tensor
    conditional: Mapping[str, Tensor]


def _encode_targets(
    route_ids: Sequence[str],
    *,
    axes: HierarchicalRouterAxes,
    split: str,
    require_all_routes: bool = True,
) -> _EncodedTargets:
    if not route_ids:
        raise ValueError(f"{split} routes must not be empty")
    joint_to_index = {route_id: index for index, route_id in enumerate(axes.joint_routes)}
    root_to_index = {route_id: index for index, route_id in enumerate(axes.root_routes)}
    leaf_to_family = axes.leaf_to_family
    family_leaf_to_index = {
        family_id: {leaf_id: index for index, leaf_id in enumerate(leaves)}
        for family_id, leaves in axes.family_leaves
    }
    flat: list[int] = []
    root: list[int] = []
    conditional = {family_id: [] for family_id, _leaves in axes.family_leaves}
    counts = {route_id: 0 for route_id in axes.joint_routes}
    for row, route_id in enumerate(route_ids):
        if not isinstance(route_id, str) or route_id not in joint_to_index:
            raise ValueError(f"{split} row {row} has unknown route_id {route_id!r}")
        counts[route_id] += 1
        flat.append(joint_to_index[route_id])
        if route_id == BASE_ROUTE:
            root.append(root_to_index[BASE_ROUTE])
            for family_id in conditional:
                conditional[family_id].append(-1)
            continue
        family_id = leaf_to_family[route_id]
        root.append(root_to_index[family_id])
        for candidate_family in conditional:
            conditional[candidate_family].append(
                family_leaf_to_index[candidate_family][route_id]
                if candidate_family == family_id
                else -1
            )
    missing = [route_id for route_id, count in counts.items() if count == 0]
    if require_all_routes and missing:
        raise ValueError(f"{split} has no examples for routes: {missing!r}")
    return _EncodedTargets(
        flat=torch.tensor(flat, dtype=torch.long),
        root=torch.tensor(root, dtype=torch.long),
        conditional={
            family_id: torch.tensor(values, dtype=torch.long)
            for family_id, values in conditional.items()
        },
    )


def _balanced_weights(labels: Tensor, classes: int) -> Tensor:
    counts = torch.bincount(labels, minlength=classes).float()
    if torch.any(counts == 0):
        raise ValueError("balanced loss requires support for every class")
    inverse = counts.sum() / (classes * counts)
    return inverse / inverse.mean()


def _regularizers(logits: Tensor) -> tuple[Tensor, Tensor]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    assignments = F.one_hot(probabilities.argmax(dim=-1), logits.shape[-1]).float()
    load_balance = logits.shape[-1] * torch.sum(
        assignments.mean(dim=0).detach() * probabilities.mean(dim=0)
    )
    z_loss = torch.logsumexp(logits.float(), dim=-1).square().mean()
    return load_balance, z_loss


def _objective(
    output: HierarchicalRouterOutput,
    targets: _EncodedTargets,
    indices: Tensor,
    *,
    root_weights: Tensor,
    leaf_weights: Mapping[str, Tensor],
    training: RouterTrainingConfig,
) -> Tensor:
    batch_root = targets.root[indices].to(output.root_logits.device)
    classification = F.cross_entropy(
        output.root_logits.float(),
        batch_root,
        weight=root_weights.to(output.root_logits.device),
        label_smoothing=float(training.label_smoothing),
    )
    load_balance, z_loss = _regularizers(output.root_logits)
    conditional_sum = output.root_logits.new_zeros((), dtype=torch.float32)
    conditional_rows = 0
    conditional_balance = output.root_logits.new_zeros((), dtype=torch.float32)
    conditional_z = output.root_logits.new_zeros((), dtype=torch.float32)
    active_heads = 0
    for family_id, logits in output.conditional_leaf_logits.items():
        labels = targets.conditional[family_id][indices].to(logits.device)
        mask = labels >= 0
        count = int(mask.sum().item())
        if count == 0:
            continue
        conditional_sum = conditional_sum + F.cross_entropy(
            logits[mask].float(),
            labels[mask],
            weight=leaf_weights[family_id].to(logits.device),
            label_smoothing=float(training.label_smoothing),
            reduction="sum",
        )
        head_balance, head_z = _regularizers(logits[mask])
        conditional_balance = conditional_balance + head_balance
        conditional_z = conditional_z + head_z
        conditional_rows += count
        active_heads += 1
    if conditional_rows:
        classification = classification + conditional_sum / conditional_rows
    if active_heads:
        load_balance = load_balance + conditional_balance / active_heads
        z_loss = z_loss + conditional_z / active_heads
    return (
        classification
        + training.load_balance_weight * load_balance
        + training.z_loss_weight * z_loss
    )


def _calibration_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, Any]:
    predictions = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    accuracy = (predictions == labels).float()
    one_hot = F.one_hot(labels, probabilities.shape[-1]).float()
    brier = (probabilities - one_hot).square().sum(dim=-1).mean()
    ece = probabilities.new_zeros(())
    bins: list[dict[str, Any]] = []
    for lower_index in range(10):
        lower = lower_index / 10
        upper = (lower_index + 1) / 10
        mask = (confidence >= lower) & (confidence <= upper if upper == 1.0 else confidence < upper)
        support = int(mask.sum().item())
        if support:
            bin_accuracy = accuracy[mask].mean()
            bin_confidence = confidence[mask].mean()
            ece = ece + mask.float().mean() * (bin_accuracy - bin_confidence).abs()
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "support": support,
                    "accuracy": float(bin_accuracy.item()),
                    "confidence": float(bin_confidence.item()),
                }
            )
    selective_risk: dict[str, Any] = {}
    for threshold in (0.0, 0.5, 0.7, 0.9):
        selected = confidence >= threshold
        support = int(selected.sum().item())
        selective_risk[f"{threshold:.1f}"] = {
            "coverage": support / len(labels),
            "risk": float(1 - accuracy[selected].mean().item()) if support else None,
        }
    return {
        "brier": float(brier.item()),
        "ece": float(ece.item()),
        "bins": bins,
        "selective_risk": selective_risk,
    }


def _same_runtime_device(left: torch.device, right: torch.device) -> bool:
    """Treat an unindexed CUDA device as the process's current CUDA device."""

    if left.type != right.type:
        return False
    if left.type != "cuda":
        return left == right
    current_index = torch.cuda.current_device()
    left_index = current_index if left.index is None else left.index
    right_index = current_index if right.index is None else right.index
    return left_index == right_index


def evaluate_hierarchical_router(
    router: HierarchicalRouterHeads,
    features: Tensor,
    route_ids: Sequence[str],
    *,
    batch_size: int,
    device: torch.device,
    require_all_routes: bool = True,
) -> dict[str, Any]:
    """Measure root, conditional, and final joint-route behavior."""

    if features.ndim != 2 or features.shape[1] != router.hidden_size:
        raise ValueError(f"features must have shape [rows, {router.hidden_size}]")
    if len(route_ids) != features.shape[0]:
        raise ValueError("evaluation routes must align with features")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(require_all_routes, bool):
        raise TypeError("require_all_routes must be boolean")
    targets = _encode_targets(
        route_ids,
        axes=router.axes,
        split="evaluation",
        require_all_routes=require_all_routes,
    )
    parameter_device = next(router.parameters()).device
    if not _same_runtime_device(parameter_device, device):
        raise ValueError(
            f"router parameters are on {parameter_device}, but evaluation requested {device}"
        )
    router.eval()
    probability_rows: list[Tensor] = []
    root_correct = 0
    conditional_correct = 0
    conditional_total = 0
    row_indices = torch.arange(features.shape[0], dtype=torch.long)
    with torch.inference_mode():
        for batch_features, batch_indices in DataLoader(
            TensorDataset(features, row_indices), batch_size=batch_size, shuffle=False
        ):
            output = router(batch_features.to(device=device, dtype=torch.float32))
            probability_rows.append(router._joint_probabilities_from_output(output).cpu())
            batch_root = targets.root[batch_indices]
            root_correct += int(
                (output.root_logits.argmax(dim=-1).cpu() == batch_root).sum().item()
            )
            for family_id, logits in output.conditional_leaf_logits.items():
                labels_for_family = targets.conditional[family_id][batch_indices]
                mask = labels_for_family >= 0
                conditional_correct += int(
                    (logits.argmax(dim=-1).cpu()[mask] == labels_for_family[mask]).sum().item()
                )
                conditional_total += int(mask.sum().item())
    probabilities = torch.cat(probability_rows, dim=0)
    predictions = probabilities.argmax(dim=-1)
    labels = targets.flat
    route_count = len(router.axes.joint_routes)
    confusion = torch.zeros((route_count, route_count), dtype=torch.long)
    for expected, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[expected, predicted] += 1
    safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    row_nll = -safe_probabilities[torch.arange(len(labels)), labels].log()
    per_route: dict[str, Any] = {}
    recalls: list[float] = []
    supported_route_nlls: list[float] = []
    unsupported_routes: list[str] = []
    for index, route_id in enumerate(router.axes.joint_routes):
        support = int(confusion[index].sum().item())
        predicted = int(confusion[:, index].sum().item())
        true_positive = int(confusion[index, index].item())
        recall = true_positive / support if support else None
        route_nll = float(row_nll[labels == index].mean().item()) if support else None
        if recall is None:
            unsupported_routes.append(route_id)
        else:
            recalls.append(recall)
            assert route_nll is not None
            supported_route_nlls.append(route_nll)
        per_route[route_id] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": recall,
            "nll": route_nll,
        }
    result = {
        "loss": float(F.nll_loss(safe_probabilities.log(), labels).item()),
        "supported_route_macro_nll": sum(supported_route_nlls) / len(supported_route_nlls),
        "accuracy": float((predictions == labels).float().mean().item()),
        "macro_recall": sum(recalls) / len(recalls),
        "unsupported_routes": unsupported_routes,
        "root_accuracy": root_correct / len(route_ids),
        "conditional_leaf_accuracy": (
            conditional_correct / conditional_total if conditional_total else None
        ),
        "mean_confidence": float(probabilities.max(dim=-1).values.mean().item()),
        "examples": len(route_ids),
        "confusion": confusion.tolist(),
        "per_route": per_route,
    }
    result.update(_calibration_metrics(probabilities, labels))
    return result


def _training_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _normalized_prompt_fingerprint(prompt: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", prompt).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _validate_dataset_isolation(
    datasets: Mapping[str, Sequence[RouterExample]],
) -> None:
    by_split: dict[str, set[str]] = {}
    for split, examples in datasets.items():
        fingerprints = [_normalized_prompt_fingerprint(example.prompt) for example in examples]
        duplicate_count = len(fingerprints) - len(set(fingerprints))
        if duplicate_count:
            raise ValueError(
                f"{split} contains {duplicate_count} duplicate normalized router prompts"
            )
        by_split[split] = set(fingerprints)
    splits = tuple(by_split)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            overlap = by_split[left].intersection(by_split[right])
            if overlap:
                raise ValueError(
                    f"router prompt leakage between {left} and {right}: {len(overlap)} rows"
                )


def _enable_deterministic_training(device: torch.device) -> dict[str, Any]:
    """Fail closed unless torch is configured for repeatable router optimization."""

    cublas_workspace: str | None = None
    if device.type == "cuda":
        cublas_workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if cublas_workspace not in {":16:8", ":4096:8"}:
            raise RuntimeError("CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:16:8 or :4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "cublas_workspace_config": cublas_workspace,
        "tf32_matmul": (
            bool(torch.backends.cuda.matmul.allow_tf32) if device.type == "cuda" else None
        ),
    }


def fit_hierarchical_router(
    train_features: Tensor,
    train_route_ids: Sequence[str],
    validation_features: Tensor,
    validation_route_ids: Sequence[str],
    *,
    taxonomy: ExpertTaxonomy,
    training: RouterTrainingConfig = RouterTrainingConfig(),
) -> tuple[HierarchicalRouterHeads, dict[str, Any]]:
    """Fit class-balanced heads and retain the best supported-route macro validation NLL."""

    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)
    train_targets = _encode_targets(train_route_ids, axes=axes, split="train")
    validation_targets = _encode_targets(validation_route_ids, axes=axes, split="validation")
    for split, features, targets in (
        ("train", train_features, train_targets),
        ("validation", validation_features, validation_targets),
    ):
        if features.ndim != 2 or not features.is_floating_point():
            raise ValueError(f"{split} features must be a rank-2 floating tensor")
        if features.shape[1] < 1:
            raise ValueError(f"{split} features must have positive width")
        if features.shape[0] != len(targets.flat):
            raise ValueError(f"{split} routes must align with features")
        if not torch.isfinite(features).all().item():
            raise ValueError(f"{split} features must contain only finite values")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("train and validation feature widths must match")

    random.seed(training.seed)
    torch.manual_seed(training.seed)
    device = _training_device(training.device)
    determinism = _enable_deterministic_training(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    router = HierarchicalRouterHeads(
        hidden_size=train_features.shape[1],
        axes=axes,
        router_hidden_size=training.router_hidden_size,
    ).to(device)
    root_weights = _balanced_weights(train_targets.root, len(axes.root_routes))
    leaf_weights = {
        family_id: _balanced_weights(
            labels[labels >= 0],
            len(dict(axes.family_leaves)[family_id]),
        )
        for family_id, labels in train_targets.conditional.items()
    }
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(training.seed)
    row_indices = torch.arange(train_features.shape[0], dtype=torch.long)
    loader = DataLoader(
        TensorDataset(train_features.float(), row_indices),
        batch_size=training.batch_size,
        shuffle=True,
        generator=generator,
    )
    selection_metric = "validation.supported_route_macro_nll"
    best_selection_value = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, training.epochs + 1):
        router.train()
        running = 0.0
        seen = 0
        for batch_features, batch_indices in loader:
            batch_features = batch_features.to(device)
            output = router(batch_features)
            loss = _objective(
                output,
                train_targets,
                batch_indices,
                root_weights=root_weights,
                leaf_weights=leaf_weights,
                training=training,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(batch_indices)
            running += float(loss.detach().item()) * count
            seen += count
        validation_metrics = evaluate_hierarchical_router(
            router,
            validation_features.float(),
            validation_route_ids,
            batch_size=training.batch_size,
            device=device,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_objective": running / seen,
            "validation_loss": validation_metrics["loss"],
            "validation_supported_route_macro_nll": validation_metrics["supported_route_macro_nll"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_recall": validation_metrics["macro_recall"],
        }
        history.append(epoch_metrics)
        selection_value = float(validation_metrics["supported_route_macro_nll"])
        if selection_value < best_selection_value - 1e-8:
            best_selection_value = selection_value
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone() for name, value in router.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= training.patience:
                break
    if best_state is None:
        raise RuntimeError("hierarchical router training did not produce a checkpoint")
    router.load_state_dict(best_state)
    router.to(device)
    metrics = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "selection": {
            "metric": selection_metric,
            "mode": "min",
            "best_value": best_selection_value,
            "minimum_improvement": 1e-8,
        },
        "history": history,
        "train": evaluate_hierarchical_router(
            router,
            train_features.float(),
            train_route_ids,
            batch_size=training.batch_size,
            device=device,
        ),
        "validation": evaluate_hierarchical_router(
            router,
            validation_features.float(),
            validation_route_ids,
            batch_size=training.batch_size,
            device=device,
        ),
        "class_balance": {
            "strategy": "inverse_frequency_per_head",
            "root_weights": root_weights.tolist(),
            "leaf_weights": {
                family_id: weights.tolist() for family_id, weights in leaf_weights.items()
            },
        },
        "determinism": {"seed": training.seed, **determinism},
    }
    return router.cpu(), metrics


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path, *, logical_name: str | None = None) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"artifact does not exist: {source}")
    before = source.stat()
    sha256 = _sha256_file(source)
    after = source.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError(f"artifact changed while it was hashed: {source}")
    return {
        "logical_name": logical_name or source.name,
        "path": str(source),
        "sha256": sha256,
        "size_bytes": after.st_size,
    }


def _tokenizer_artifacts(model_dir: Path) -> list[dict[str, Any]]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    )
    return [_artifact(model_dir / name) for name in names if (model_dir / name).is_file()]


def _leaf_adapter_artifacts(
    adapters: Mapping[str, str | Path], axes: HierarchicalRouterAxes
) -> list[dict[str, Any]]:
    expected = set(axes.joint_routes[1:])
    if set(adapters) != expected:
        missing = sorted(expected - set(adapters))
        unknown = sorted(set(adapters) - expected)
        raise ValueError(
            f"adapters must map every canonical leaf exactly (missing={missing}, unknown={unknown})"
        )
    results: list[dict[str, Any]] = []
    for leaf_id in axes.joint_routes[1:]:
        directory = Path(adapters[leaf_id]).expanduser().resolve()
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"adapter leaf is incomplete: {directory}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adapter config: {config_path}") from exc
        if config.get("name") != leaf_id:
            raise ValueError(
                f"adapter leaf {leaf_id!r} differs from saved name {config.get('name')!r}"
            )
        results.append(
            {
                "route_id": leaf_id,
                "path": str(directory),
                "config": _artifact(config_path),
                "weights": _artifact(weights_path),
            }
        )
    return results


def _strict_json_object(path: Path, field: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            result[key] = value
        return result

    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != before.st_size
    ):
        raise RuntimeError(f"{field} changed while it was read")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{field} contains unsupported numeric constant {item!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return value


def _verify_artifact_record(
    record: object,
    path: Path,
    *,
    field: str,
    logical_name: str,
) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "logical_name",
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError(f"{field} must be an exact artifact record")
    if record["logical_name"] != logical_name:
        raise ValueError(f"{field} logical name differs from {logical_name!r}")
    if (
        not isinstance(record["size_bytes"], int)
        or isinstance(record["size_bytes"], bool)
        or record["size_bytes"] < 0
    ):
        raise ValueError(f"{field}.size_bytes must be a non-negative integer")
    claimed = validate_sha256(record["sha256"], f"{field}.sha256")
    if record["sha256"] != claimed:
        raise ValueError(f"{field}.sha256 must use lowercase hexadecimal")
    observed = _artifact(path, logical_name=logical_name)
    if observed["sha256"] != claimed or observed["size_bytes"] != record["size_bytes"]:
        raise ValueError(f"{field} differs from the pinned artifact")


def _normalized_model_id(value: str | Path) -> str:
    return os.path.normcase(str(Path(value).expanduser().resolve()))


def load_hierarchical_router_checkpoint(
    checkpoint_directory: str | Path,
    *,
    taxonomy_path: str | Path,
    adapters: Mapping[str, str | Path] | None = None,
    expected_model_id: str | Path | None = None,
    verify_adapter_artifacts: bool = True,
    verify_model_artifacts: bool = True,
    device: str | torch.device = "cpu",
) -> HierarchicalRouterCheckpoint:
    """Load a taxonomy-bound checkpoint after verifying every serving-critical contract."""

    directory = Path(checkpoint_directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"router checkpoint directory does not exist: {directory}")
    config_path = directory / "router_config.json"
    weights_path = directory / "router.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "hierarchical checkpoint needs router_config.json and router.safetensors"
        )
    payload = _strict_json_object(config_path, "router_config.json")
    if payload.get("schema_version") != HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported hierarchical router checkpoint schema")
    claimed_content = validate_sha256(payload.get("content_sha256"), "router_config.content_sha256")
    if payload.get("content_sha256") != claimed_content:
        raise ValueError("router_config.content_sha256 must use lowercase hexadecimal")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    if fingerprint_json(semantic) != claimed_content:
        raise ValueError("router_config.json semantic fingerprint drift")

    taxonomy_source = Path(taxonomy_path).expanduser().resolve()
    taxonomy = load_expert_taxonomy(taxonomy_source)
    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)
    raw_taxonomy = payload.get("taxonomy")
    if not isinstance(raw_taxonomy, Mapping):
        raise ValueError("router checkpoint has no taxonomy contract")
    if (
        raw_taxonomy.get("taxonomy_id") != taxonomy.taxonomy_id
        or raw_taxonomy.get("revision") != taxonomy.revision
        or raw_taxonomy.get("content_sha256") != taxonomy.fingerprint
    ):
        raise ValueError("router checkpoint targets a different taxonomy")
    _verify_artifact_record(
        raw_taxonomy.get("artifact"),
        taxonomy_source,
        field="taxonomy.artifact",
        logical_name=taxonomy_source.name,
    )
    if payload.get("axes") != axes.to_dict():
        raise ValueError("router checkpoint axes differ from the canonical taxonomy order")

    raw_head = payload.get("head_config")
    if not isinstance(raw_head, Mapping) or set(raw_head) != {
        "hidden_size",
        "router_hidden_size",
        "root_classes",
        "conditional_classes",
    }:
        raise ValueError("router checkpoint has an invalid head_config")
    hidden_size = raw_head["hidden_size"]
    router_hidden_size = raw_head["router_hidden_size"]
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
        raise ValueError("head_config.hidden_size must be a positive integer")
    if router_hidden_size is not None and (
        not isinstance(router_hidden_size, int)
        or isinstance(router_hidden_size, bool)
        or router_hidden_size < 1
    ):
        raise ValueError("head_config.router_hidden_size must be positive or null")
    expected_conditionals = {family_id: len(leaves) for family_id, leaves in axes.family_leaves}
    if (
        raw_head["root_classes"] != len(axes.root_routes)
        or raw_head["conditional_classes"] != expected_conditionals
    ):
        raise ValueError("router head class counts differ from the taxonomy axes")

    raw_feature = payload.get("feature_contract")
    if not isinstance(raw_feature, dict):
        raise ValueError("router checkpoint has no feature_contract")
    try:
        feature_contract = RouterFeatureContract(**raw_feature)
    except (TypeError, ValueError) as exc:
        raise ValueError("router checkpoint has an invalid feature_contract") from exc
    raw_model = payload.get("model")
    if not isinstance(raw_model, Mapping):
        raise ValueError("router checkpoint has no frozen-base model contract")
    model_id = raw_model.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("router checkpoint has no frozen-base model identity")
    if raw_model.get("hidden_size") != hidden_size:
        raise ValueError("router and frozen-base hidden sizes differ")
    if expected_model_id is not None and _normalized_model_id(model_id) != _normalized_model_id(
        expected_model_id
    ):
        raise ValueError("router checkpoint targets a different frozen base model")
    if verify_model_artifacts:
        model_root = Path(expected_model_id or model_id).expanduser().resolve()
        if not model_root.is_dir():
            raise FileNotFoundError(f"frozen-base model directory does not exist: {model_root}")
        _verify_artifact_record(
            raw_model.get("config"),
            model_root / "config.json",
            field="model.config",
            logical_name="config.json",
        )
        embedding = raw_model.get("embedding")
        if not isinstance(embedding, Mapping):
            raise ValueError("router checkpoint has no input-embedding contract")
        shape = embedding.get("shape")
        if (
            embedding.get("tensor_name") != feature_contract.embedding_tensor
            or not isinstance(shape, list)
            or len(shape) != 2
            or shape[1] != hidden_size
            or not isinstance(embedding.get("dtype"), str)
        ):
            raise ValueError("input-embedding contract differs from router features")
        embedding_record = embedding.get("artifact")
        if not isinstance(embedding_record, Mapping):
            raise ValueError("input-embedding artifact record is missing")
        embedding_name = embedding_record.get("logical_name")
        if not isinstance(embedding_name, str) or Path(embedding_name).name != embedding_name:
            raise ValueError("input-embedding artifact logical name is unsafe")
        _verify_artifact_record(
            embedding_record,
            model_root / embedding_name,
            field="model.embedding.artifact",
            logical_name=embedding_name,
        )
        tokenizer_records = raw_model.get("tokenizer_artifacts")
        if not isinstance(tokenizer_records, list) or not tokenizer_records:
            raise ValueError("router checkpoint has no tokenizer artifact contract")
        tokenizer_names: list[str] = []
        for index, record in enumerate(tokenizer_records):
            if not isinstance(record, Mapping):
                raise ValueError(f"model.tokenizer_artifacts[{index}] must be an object")
            logical_name = record.get("logical_name")
            if not isinstance(logical_name, str) or Path(logical_name).name != logical_name:
                raise ValueError("tokenizer artifact logical name is unsafe")
            tokenizer_names.append(logical_name)
            _verify_artifact_record(
                record,
                model_root / logical_name,
                field=f"model.tokenizer_artifacts[{index}]",
                logical_name=logical_name,
            )
        if len(tokenizer_names) != len(set(tokenizer_names)):
            raise ValueError("tokenizer artifact contract contains duplicate logical names")

    raw_adapters = payload.get("adapters")
    if not isinstance(raw_adapters, list):
        raise ValueError("router checkpoint has no adapter contract")
    adapter_ids = tuple(
        item.get("route_id") if isinstance(item, Mapping) else None for item in raw_adapters
    )
    if adapter_ids != axes.joint_routes[1:]:
        raise ValueError("router checkpoint adapter order differs from the taxonomy leaf axis")
    if adapters is not None and set(adapters) != set(adapter_ids):
        raise ValueError("runtime adapters must map every canonical leaf exactly")
    if verify_adapter_artifacts:
        for item in raw_adapters:
            assert isinstance(item, Mapping)
            route_id = str(item["route_id"])
            directory_value = adapters[route_id] if adapters is not None else item.get("path")
            if not isinstance(directory_value, (str, Path)):
                raise ValueError(f"adapter {route_id!r} has no usable directory")
            adapter_dir = Path(directory_value).expanduser().resolve()
            config = adapter_dir / "adapter_config.json"
            weights = adapter_dir / "adapter_model.safetensors"
            _verify_artifact_record(
                item.get("config"),
                config,
                field=f"adapters[{route_id!r}].config",
                logical_name="adapter_config.json",
            )
            _verify_artifact_record(
                item.get("weights"),
                weights,
                field=f"adapters[{route_id!r}].weights",
                logical_name="adapter_model.safetensors",
            )
            adapter_config = _strict_json_object(config, f"adapter {route_id!r} config")
            if adapter_config.get("name") != route_id:
                raise ValueError(f"adapter {route_id!r} saved name differs from its route")

    _verify_artifact_record(
        payload.get("checkpoint"),
        weights_path,
        field="checkpoint",
        logical_name="router.safetensors",
    )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to load a hierarchical router") from exc
    tensors = load_file(str(weights_path), device="cpu")
    if not tensors or any(not tensor.is_floating_point() for tensor in tensors.values()):
        raise ValueError("router.safetensors must contain floating-point head tensors")
    if any(not torch.isfinite(tensor).all().item() for tensor in tensors.values()):
        raise ValueError("router.safetensors contains non-finite parameters")
    router = HierarchicalRouterHeads(
        hidden_size=hidden_size,
        axes=axes,
        router_hidden_size=router_hidden_size,
    )
    try:
        router.load_state_dict(tensors, strict=True)
    except RuntimeError as exc:
        raise ValueError("router.safetensors tensor schema differs from head_config") from exc
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    router.to(device=target_device, dtype=torch.float32)
    router.requires_grad_(False)
    router.eval()
    return HierarchicalRouterCheckpoint(
        directory=directory,
        router=router,
        axes=axes,
        feature_contract=feature_contract,
        model_id=model_id,
        adapter_ids=adapter_ids,  # type: ignore[arg-type]
        sidecar=MappingProxyType(payload),
    )


def _seal_checkpoint_bundle(
    *,
    destination: Path,
    router: HierarchicalRouterHeads,
    sidecar: Mapping[str, Any],
    taxonomy_path: Path,
    adapters: Mapping[str, str | Path],
    model_id: Path,
) -> tuple[Path, Path]:
    if "checkpoint" in sidecar or "content_sha256" in sidecar:
        raise ValueError("unsealed sidecar must not predeclare checkpoint identity")
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to save a hierarchical router") from exc

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    staged_checkpoint = staging / "router.safetensors"
    staged_config = staging / "router_config.json"
    try:
        save_file(
            {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in router.state_dict().items()
            },
            staged_checkpoint,
        )
        checkpoint_artifact = _artifact(staged_checkpoint, logical_name="router.safetensors")
        checkpoint_artifact["path"] = "router.safetensors"
        sealed = dict(sidecar)
        sealed["checkpoint"] = checkpoint_artifact
        sealed["content_sha256"] = fingerprint_json(sealed)
        staged_config.write_text(
            json.dumps(sealed, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        load_hierarchical_router_checkpoint(
            staging,
            taxonomy_path=taxonomy_path,
            adapters=adapters,
            expected_model_id=model_id,
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"hierarchical router output appeared during training: {destination}"
            )
        os.replace(staging, destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination / "router.safetensors", destination / "router_config.json"


def train_hierarchical_router_checkpoint(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    taxonomy_path: str | Path,
    model_dir: str | Path,
    adapters: Mapping[str, str | Path],
    output_dir: str | Path,
    feature_contract: RouterFeatureContract = RouterFeatureContract(),
    training: RouterTrainingConfig = RouterTrainingConfig(),
    feature_batch_size: int = 64,
    cache_dir: str | Path | None = None,
    limit_per_route: int | None = None,
    progress: ProgressCallback | None = None,
) -> HierarchicalRouterTrainingResult:
    """Extract frozen features, train the hierarchy, and seal its adapter-bound checkpoint."""

    if (
        not isinstance(feature_batch_size, int)
        or isinstance(feature_batch_size, bool)
        or feature_batch_size < 1
    ):
        raise ValueError("feature_batch_size must be a positive integer")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to replace an existing hierarchical router artifact: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    taxonomy_source = Path(taxonomy_path).resolve()
    taxonomy = load_expert_taxonomy(taxonomy_source)
    axes = HierarchicalRouterAxes.from_taxonomy(taxonomy)
    source_paths: dict[str, Path] = {
        "train": Path(train_path).resolve(),
        "validation": Path(validation_path).resolve(),
    }
    if test_path is not None:
        source_paths["test"] = Path(test_path).resolve()
    datasets = {
        split: load_router_jsonl(
            path,
            route_ids=axes.joint_routes,
            limit_per_route=limit_per_route,
            require_all_routes=split != "test",
        )
        for split, path in source_paths.items()
    }
    _validate_dataset_isolation(datasets)

    model_root = Path(model_dir).expanduser().resolve()
    model_config_path = model_root / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {model_config_path}")
    model_config = _strict_json_object(model_config_path, "model config")
    hidden_size = model_config.get("hidden_size")
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1:
        raise ValueError(f"model config has no valid hidden_size: {model_config_path}")
    architectures = model_config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list)
        or any(not isinstance(item, str) or not item for item in architectures)
    ):
        raise ValueError("model config architectures must be an array of non-empty strings")
    embedding, embedding_metadata = load_input_embedding(
        model_root, tensor_name=feature_contract.embedding_tensor
    )
    if embedding.shape[1] != hidden_size:
        raise ValueError("model hidden_size differs from the input embedding width")
    tokenizer = load_router_tokenizer(model_root)
    cache_root = None if cache_dir is None else Path(cache_dir).resolve()
    split_features: dict[str, Tensor] = {}
    cache_hits: dict[str, bool] = {}
    for split, examples in datasets.items():
        cache_path = None if cache_root is None else cache_root / f"{split}.safetensors"
        features, _flat_labels, hit = cached_prompt_features(
            source_paths[split],
            examples=examples,
            tokenizer=tokenizer,
            embedding=embedding,
            embedding_metadata=embedding_metadata,
            route_ids=axes.joint_routes,
            contract=feature_contract,
            cache_path=cache_path,
            batch_size=feature_batch_size,
            progress=(
                None if progress is None else lambda text, name=split: progress(f"{name}: {text}")
            ),
        )
        split_features[split] = features
        cache_hits[split] = hit
    del embedding

    train_routes = [example.route_id for example in datasets["train"]]
    validation_routes = [example.route_id for example in datasets["validation"]]
    router, metrics = fit_hierarchical_router(
        split_features["train"],
        train_routes,
        split_features["validation"],
        validation_routes,
        taxonomy=taxonomy,
        training=training,
    )
    if "test" in datasets:
        metrics["test"] = evaluate_hierarchical_router(
            router,
            split_features["test"],
            [example.route_id for example in datasets["test"]],
            batch_size=training.batch_size,
            device=torch.device("cpu"),
            require_all_routes=False,
        )
    metrics["feature_cache_hits"] = cache_hits

    adapter_metadata = _leaf_adapter_artifacts(adapters, axes)
    sidecar = {
        "schema_version": HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA,
        "taxonomy": {
            "taxonomy_id": taxonomy.taxonomy_id,
            "revision": taxonomy.revision,
            "content_sha256": taxonomy.fingerprint,
            "artifact": _artifact(taxonomy_source),
        },
        "axes": axes.to_dict(),
        "model": {
            "model_id": str(model_root),
            "architecture": next(iter(architectures or ()), None),
            "hidden_size": hidden_size,
            "config": _artifact(model_config_path),
            "embedding": embedding_metadata,
            "tokenizer_artifacts": _tokenizer_artifacts(model_root),
        },
        "adapters": adapter_metadata,
        "feature_contract": feature_contract.to_dict(),
        "head_config": {
            "hidden_size": router.hidden_size,
            "router_hidden_size": router.router_hidden_size,
            "root_classes": len(axes.root_routes),
            "conditional_classes": {
                family_id: len(leaves) for family_id, leaves in axes.family_leaves
            },
        },
        "training_config": training.to_dict(),
        "datasets": {split: _artifact(path) for split, path in source_paths.items()},
        "metrics": metrics,
        "calibration_contract": {
            "measured": ["brier", "ece", "selective_risk"],
            "deferred": {
                "expert_quality_regret": (
                    "requires held-out base-versus-expert generation quality measurements"
                )
            },
        },
    }
    checkpoint_path, config_path = _seal_checkpoint_bundle(
        destination=destination,
        router=router,
        sidecar=sidecar,
        taxonomy_path=taxonomy_source,
        adapters=adapters,
        model_id=model_root,
    )
    return HierarchicalRouterTrainingResult(
        output_dir=destination,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        metrics=metrics,
    )


__all__ = [
    "HIERARCHICAL_ROUTER_CHECKPOINT_SCHEMA",
    "HierarchicalRouterAxes",
    "HierarchicalRouterCheckpoint",
    "HierarchicalRouterHeads",
    "HierarchicalRouterOutput",
    "HierarchicalRouterTrainingResult",
    "evaluate_hierarchical_router",
    "fit_hierarchical_router",
    "load_hierarchical_router_checkpoint",
    "train_hierarchical_router_checkpoint",
]
