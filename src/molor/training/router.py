"""Lightweight prompt-router training over frozen input-token embeddings.

The bootstrap router intentionally does not load the causal language model.  It reads the
Qwen input-embedding tensor directly from safetensors, mean-pools the tokens in each user
prompt, and trains :class:`molor.router.TopKRouter` on those cached 2048-wide features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from molor.config import RouterConfig
from molor.identity import fingerprint_json
from molor.router import TopKRouter


ROUTER_CHECKPOINT_SCHEMA = "molor.experimental-router-checkpoint.v1"
FEATURE_CACHE_SCHEMA = "molor.router-feature-cache.v1"
DEFAULT_ROUTE_IDS = (
    "embedded-knowledge-v1",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "base/null",
)


@dataclass(frozen=True, slots=True)
class RouterExample:
    """One source-label bootstrap example."""

    prompt: str
    route_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("router prompt must be a non-empty string")
        if not isinstance(self.route_id, str) or not self.route_id:
            raise ValueError("route_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RouterFeatureContract:
    """Exact prompt representation expected by training and serving."""

    max_tokens: int = 256
    source: str = "input_embeddings"
    pooling: str = "masked_mean"
    prompt_scope: str = "user_content_only"
    padding: str = "longest"
    truncation: str = "right"
    add_special_tokens: bool = True
    embedding_tensor: str = "model.embed_tokens.weight"

    def __post_init__(self) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise TypeError("max_tokens must be an integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        expected = {
            "source": "input_embeddings",
            "pooling": "masked_mean",
            "prompt_scope": "user_content_only",
            "padding": "longest",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} must be {expected_value!r} for router v1")
        if self.truncation not in {"right", "head_tail"}:
            raise ValueError("truncation must be 'right' or 'head_tail'")
        if not isinstance(self.add_special_tokens, bool):
            raise TypeError("add_special_tokens must be boolean")
        if not self.embedding_tensor:
            raise ValueError("embedding_tensor must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pooling": self.pooling,
            "prompt_scope": self.prompt_scope,
            "max_tokens": self.max_tokens,
            "padding": self.padding,
            "truncation": self.truncation,
            "add_special_tokens": self.add_special_tokens,
            "embedding_tensor": self.embedding_tensor,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RouterTrainingConfig:
    """Small-MLP optimization settings independent of the frozen base."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    load_balance_weight: float = 0.01
    z_loss_weight: float = 1e-4
    patience: int = 4
    seed: int = 83
    device: str = "auto"
    router_hidden_size: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("epochs", "batch_size", "patience"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in ("learning_rate", "weight_decay", "load_balance_weight", "z_loss_weight"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if (
            isinstance(self.label_smoothing, bool)
            or not isinstance(self.label_smoothing, (int, float))
            or not 0 <= self.label_smoothing < 1
        ):
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.device != "auto" and not self.device:
            raise ValueError("device must be 'auto' or a torch device string")
        if self.router_hidden_size is not None and (
            not isinstance(self.router_hidden_size, int)
            or isinstance(self.router_hidden_size, bool)
            or self.router_hidden_size < 1
        ):
            raise ValueError("router_hidden_size must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "label_smoothing": self.label_smoothing,
            "load_balance_weight": self.load_balance_weight,
            "z_loss_weight": self.z_loss_weight,
            "patience": self.patience,
            "seed": self.seed,
            "device": self.device,
            "router_hidden_size": self.router_hidden_size,
        }


@dataclass(frozen=True, slots=True)
class RouterTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    metrics: Mapping[str, Any]


ProgressCallback = Callable[[str], None]


def load_router_jsonl(
    path: str | Path,
    *,
    route_ids: Sequence[str] = DEFAULT_ROUTE_IDS,
    limit_per_route: int | None = None,
    require_all_routes: bool = True,
) -> list[RouterExample]:
    """Load strict ``prompt`` + ``route_id`` JSONL in stable file order."""

    source = Path(path)
    allowed = tuple(route_ids)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("route_ids must be non-empty and unique")
    if limit_per_route is not None and (
        not isinstance(limit_per_route, int)
        or isinstance(limit_per_route, bool)
        or limit_per_route < 1
    ):
        raise ValueError("limit_per_route must be a positive integer or None")
    if not isinstance(require_all_routes, bool):
        raise TypeError("require_all_routes must be boolean")
    counts = {route_id: 0 for route_id in allowed}
    examples: list[RouterExample] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or set(raw) != {"prompt", "route_id"}:
                    raise ValueError("row must contain exactly prompt and route_id")
                example = RouterExample(prompt=raw["prompt"], route_id=raw["route_id"])
                if example.route_id not in counts:
                    raise ValueError(f"unknown route_id {example.route_id!r}")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            if limit_per_route is not None and counts[example.route_id] >= limit_per_route:
                continue
            examples.append(example)
            counts[example.route_id] += 1
    if not examples:
        raise ValueError(f"{source} contains no router examples")
    missing = [route_id for route_id, count in counts.items() if count == 0]
    if require_all_routes and missing:
        raise ValueError(f"{source} has no examples for routes: {missing!r}")
    return examples


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, logical_name: str | None = None) -> dict[str, Any]:
    return {
        "logical_name": logical_name or path.name,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _model_embedding_artifact(
    model_dir: Path,
    tensor_name: str,
) -> tuple[Path, dict[str, Any]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_name = index["weight_map"][tensor_name]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"model index does not map embedding tensor {tensor_name!r}") from exc
        if not isinstance(shard_name, str) or Path(shard_name).name != shard_name:
            raise ValueError("embedding shard name in model index is unsafe")
        shard_path = model_dir / shard_name
    else:
        shard_path = model_dir / "model.safetensors"
    if not shard_path.is_file():
        raise FileNotFoundError(f"embedding safetensors shard does not exist: {shard_path}")
    return shard_path, _artifact(shard_path)


def load_input_embedding(
    model_dir: str | Path,
    *,
    tensor_name: str = "model.embed_tokens.weight",
) -> tuple[Tensor, dict[str, Any]]:
    """Read only the tied input-embedding tensor from a local safetensors checkpoint."""

    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to train a router") from exc

    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"router training requires a local model directory: {root}")
    shard_path, shard_artifact = _model_embedding_artifact(root, tensor_name)
    with safe_open(str(shard_path), framework="pt", device="cpu") as stream:
        if tensor_name not in stream.keys():
            raise KeyError(f"embedding tensor is absent from {shard_path.name}: {tensor_name}")
        embedding = stream.get_tensor(tensor_name)
    if embedding.ndim != 2 or not embedding.is_floating_point():
        raise ValueError("input embedding must be a rank-2 floating tensor")
    metadata = {
        "tensor_name": tensor_name,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype).removeprefix("torch."),
        "artifact": shard_artifact,
    }
    return embedding, metadata


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


def load_router_tokenizer(model_dir: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to train a router") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_dir).expanduser().resolve()),
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("router tokenizer needs a pad or EOS token")
    return tokenizer


def _tokenize_prompt_batch(
    prompts: Sequence[str],
    *,
    tokenizer: Any,
    contract: RouterFeatureContract,
) -> tuple[Tensor, Tensor]:
    """Tokenize one batch according to the sealed router feature contract."""

    if not prompts:
        raise ValueError("router prompt batch must not be empty")
    if contract.truncation == "right":
        encoded = tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=contract.max_tokens,
            add_special_tokens=contract.add_special_tokens,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer must return a mapping for router prompts")
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None and isinstance(input_ids, Tensor):
            attention_mask = torch.ones_like(input_ids)
    else:
        encoded = tokenizer(
            list(prompts),
            padding=False,
            truncation=False,
            add_special_tokens=contract.add_special_tokens,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer must return a mapping for router prompts")
        raw_input_ids = encoded.get("input_ids")
        if isinstance(raw_input_ids, Tensor):
            if raw_input_ids.ndim != 2:
                raise ValueError("tokenizer returned invalid input_ids for router prompts")
            raw_rows: Sequence[object] = raw_input_ids.tolist()
        elif isinstance(raw_input_ids, Sequence) and not isinstance(raw_input_ids, (str, bytes)):
            raw_rows = raw_input_ids
        else:
            raise ValueError("tokenizer returned invalid input_ids for router prompts")
        if len(raw_rows) != len(prompts):
            raise ValueError("tokenizer input_ids do not align with router prompts")

        head_tokens = (contract.max_tokens + 1) // 2
        tail_tokens = contract.max_tokens // 2
        retained_rows: list[list[int]] = []
        for raw_row in raw_rows:
            if isinstance(raw_row, Tensor):
                if raw_row.ndim != 1:
                    raise ValueError("tokenizer returned invalid input_ids for router prompts")
                values = raw_row.tolist()
            elif isinstance(raw_row, Sequence) and not isinstance(raw_row, (str, bytes)):
                values = list(raw_row)
            else:
                raise ValueError("tokenizer returned invalid input_ids for router prompts")
            if not values or any(
                isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in values
            ):
                raise ValueError("tokenizer returned invalid input_ids for router prompts")
            if len(values) > contract.max_tokens:
                values = values[:head_tokens] + (values[-tail_tokens:] if tail_tokens else [])
            retained_rows.append(values)

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("head_tail router tokenization requires a non-negative pad_token_id")
        width = max(len(row) for row in retained_rows)
        input_ids = torch.full((len(retained_rows), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(retained_rows), width), dtype=torch.long)
        for row_index, values in enumerate(retained_rows):
            input_ids[row_index, : len(values)] = torch.tensor(values, dtype=torch.long)
            attention_mask[row_index, : len(values)] = 1

    if (
        not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != len(prompts)
        or input_ids.shape[1] < 1
        or input_ids.shape[1] > contract.max_tokens
        or input_ids.dtype == torch.bool
        or input_ids.is_floating_point()
    ):
        raise ValueError("tokenizer returned invalid input_ids for router prompts")
    if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError("tokenizer returned an invalid attention_mask for router prompts")
    mask = attention_mask.to(device="cpu", dtype=torch.float32)
    if not torch.all((mask == 0) | (mask == 1)).item():
        raise ValueError("router attention_mask must contain only zeros and ones")
    if torch.any(mask.sum(dim=1) <= 0).item():
        raise ValueError("router attention_mask must retain at least one prompt token")
    return input_ids.to(device="cpu", dtype=torch.long), mask


def extract_prompt_features(
    examples: Sequence[RouterExample],
    *,
    tokenizer: Any,
    embedding: Tensor,
    route_ids: Sequence[str],
    contract: RouterFeatureContract,
    batch_size: int = 64,
    progress: ProgressCallback | None = None,
) -> tuple[Tensor, Tensor]:
    """Tokenize and mean-pool input embeddings without a transformer forward pass."""

    if not examples:
        raise ValueError("cannot extract features from an empty example set")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    route_to_index = {route_id: index for index, route_id in enumerate(route_ids)}
    if len(route_to_index) != len(route_ids):
        raise ValueError("route_ids must be unique")
    feature_rows: list[Tensor] = []
    labels: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids, attention_mask = _tokenize_prompt_batch(
                [example.prompt for example in batch],
                tokenizer=tokenizer,
                contract=contract,
            )
            embedding_device = embedding.device
            input_ids = input_ids.to(device=embedding_device)
            attention_mask = attention_mask.to(device=embedding_device)
            if input_ids.numel() and (
                int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= embedding.shape[0]
            ):
                raise ValueError("tokenizer vocabulary exceeds the input-embedding table")
            token_features = embedding[input_ids].float()
            denominator = attention_mask.sum(dim=1, keepdim=True)
            if torch.any(denominator == 0):
                raise ValueError("tokenizer produced an empty prompt")
            pooled = (token_features * attention_mask.unsqueeze(-1)).sum(dim=1) / denominator
            if not torch.isfinite(pooled).all().item():
                raise ValueError("frozen prompt features contain non-finite values")
            feature_rows.append(pooled.cpu())
            try:
                labels.extend(route_to_index[example.route_id] for example in batch)
            except KeyError as exc:
                raise ValueError(f"dataset contains an unknown route: {exc.args[0]!r}") from exc
            if progress is not None:
                progress(f"features {min(start + len(batch), len(examples))}/{len(examples)}")
    return torch.cat(feature_rows, dim=0).contiguous(), torch.tensor(labels, dtype=torch.long)


def cached_prompt_features(
    dataset_path: str | Path,
    *,
    examples: Sequence[RouterExample],
    tokenizer: Any,
    embedding: Tensor,
    embedding_metadata: Mapping[str, Any],
    route_ids: Sequence[str],
    contract: RouterFeatureContract,
    cache_path: str | Path | None,
    batch_size: int = 64,
    progress: ProgressCallback | None = None,
) -> tuple[Tensor, Tensor, bool]:
    """Load a content-bound feature cache or build it atomically via safetensors."""

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to cache router features") from exc

    dataset = Path(dataset_path).resolve()
    cache = None if cache_path is None else Path(cache_path).resolve()
    identity = {
        "schema_version": FEATURE_CACHE_SCHEMA,
        "dataset_sha256": sha256_file(dataset),
        "route_ids": list(route_ids),
        "feature_contract": contract.to_dict(),
        "embedding": dict(embedding_metadata),
    }
    cache_fingerprint = fingerprint_json(identity)
    if cache is not None and cache.is_file():
        try:
            with safe_open(str(cache), framework="pt", device="cpu") as stream:
                metadata = stream.metadata() or {}
                if metadata.get("cache_fingerprint") == cache_fingerprint:
                    features = stream.get_tensor("features")
                    labels = stream.get_tensor("labels")
                    if (
                        features.ndim == 2
                        and labels.ndim == 1
                        and features.shape[0] == labels.shape[0] == len(examples)
                        and features.shape[1] == embedding.shape[1]
                    ):
                        if progress is not None:
                            progress(f"feature cache hit: {cache}")
                        return features.float().contiguous(), labels.long().contiguous(), True
        except (OSError, RuntimeError):
            pass

    features, labels = extract_prompt_features(
        examples,
        tokenizer=tokenizer,
        embedding=embedding,
        route_ids=route_ids,
        contract=contract,
        batch_size=batch_size,
        progress=progress,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.tmp")
        save_file(
            {"features": features, "labels": labels},
            temporary,
            metadata={
                "schema_version": FEATURE_CACHE_SCHEMA,
                "cache_fingerprint": cache_fingerprint,
            },
        )
        temporary.replace(cache)
    return features, labels, False


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _split_metrics(
    router: TopKRouter,
    features: Tensor,
    labels: Tensor,
    *,
    route_ids: Sequence[str],
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    router.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confidence_sum = 0.0
    route_count = len(route_ids)
    confusion = [[0 for _ in range(route_count)] for _ in range(route_count)]
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            output = router(batch_features)
            logits = output.logits.float()
            probabilities = output.probabilities.float()
            predictions = logits.argmax(dim=-1)
            total_loss += F.cross_entropy(logits, batch_labels, reduction="sum").item()
            total += batch_labels.numel()
            correct += int((predictions == batch_labels).sum().item())
            confidence_sum += float(probabilities.max(dim=-1).values.sum().item())
            for expected, predicted in zip(
                batch_labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                strict=True,
            ):
                confusion[expected][predicted] += 1
    per_route: dict[str, Any] = {}
    for index, route_id in enumerate(route_ids):
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        true_positive = confusion[index][index]
        per_route[route_id] = {
            "support": support,
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / support if support else 0.0,
        }
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "mean_confidence": confidence_sum / max(total, 1),
        "examples": total,
        "confusion": confusion,
        "per_route": per_route,
    }


def fit_router(
    train_features: Tensor,
    train_labels: Tensor,
    validation_features: Tensor,
    validation_labels: Tensor,
    *,
    route_ids: Sequence[str],
    hidden_size: int,
    training: RouterTrainingConfig = RouterTrainingConfig(),
    progress: ProgressCallback | None = None,
) -> tuple[TopKRouter, dict[str, Any]]:
    """Fit a sequence-level hard-top-1 router and retain the best validation epoch."""

    routes = tuple(route_ids)
    if len(routes) < 2 or routes[-1] != "base/null":
        raise ValueError("route_ids must end with the explicit base/null route")
    for name, features, labels in (
        ("train", train_features, train_labels),
        ("validation", validation_features, validation_labels),
    ):
        if features.ndim != 2 or features.shape[1] != hidden_size:
            raise ValueError(f"{name} features must have shape [rows, {hidden_size}]")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(f"{name} labels must align with features")
        if not labels.numel():
            raise ValueError(f"{name} split must not be empty")
        if int(labels.min()) < 0 or int(labels.max()) >= len(routes):
            raise ValueError(f"{name} labels are outside the route axis")

    random.seed(training.seed)
    torch.manual_seed(training.seed)
    device = _device(training.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training.seed)
    router_config = RouterConfig(
        hidden_size=hidden_size,
        num_experts=len(routes) - 1,
        top_k=1,
        mode="sequence",
        router_hidden_size=training.router_hidden_size,
        temperature=1.0,
        include_null_expert=True,
        normalize_top_k=True,
        uniform_top_k=False,
        straight_through=False,
    )
    router = TopKRouter(router_config).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(training.seed)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=training.batch_size,
        shuffle=True,
        generator=generator,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, training.epochs + 1):
        router.train()
        running_loss = 0.0
        seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            output = router(batch_features)
            classification = F.cross_entropy(
                output.logits.float(),
                batch_labels,
                label_smoothing=float(training.label_smoothing),
            )
            loss = (
                classification
                + training.load_balance_weight * output.load_balance_loss
                + training.z_loss_weight * output.z_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = batch_labels.numel()
            running_loss += float(loss.detach().item()) * count
            seen += count
        validation_metrics = _split_metrics(
            router,
            validation_features,
            validation_labels,
            route_ids=routes,
            batch_size=training.batch_size,
            device=device,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_objective": running_loss / max(seen, 1),
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_metrics)
        if progress is not None:
            progress(
                f"epoch {epoch}/{training.epochs} train {epoch_metrics['train_objective']:.4f} "
                f"validation {epoch_metrics['validation_loss']:.4f} "
                f"accuracy {epoch_metrics['validation_accuracy']:.1%}"
            )
        if validation_metrics["loss"] < best_loss - 1e-8:
            best_loss = float(validation_metrics["loss"])
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in router.network.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= training.patience:
                break
    if best_state is None:
        raise RuntimeError("router training did not produce a checkpoint")
    router.network.load_state_dict(best_state)
    router.to(device)
    metrics = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "train": _split_metrics(
            router,
            train_features,
            train_labels,
            route_ids=routes,
            batch_size=training.batch_size,
            device=device,
        ),
        "validation": _split_metrics(
            router,
            validation_features,
            validation_labels,
            route_ids=routes,
            batch_size=training.batch_size,
            device=device,
        ),
    }
    return router.cpu(), metrics


def _adapter_artifacts(
    adapters: Mapping[str, str | Path],
    route_ids: Sequence[str],
) -> list[dict[str, Any]]:
    expected = tuple(route_ids[:-1])
    if set(adapters) != set(expected):
        raise ValueError(f"adapters must map exactly these routes: {expected!r}")
    results: list[dict[str, Any]] = []
    for route_id in expected:
        directory = Path(adapters[route_id]).expanduser().resolve()
        config_path = directory / "adapter_config.json"
        weights_path = directory / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"adapter route is incomplete: {directory}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid adapter config: {config_path}") from exc
        if config.get("name") != route_id:
            raise ValueError(
                f"adapter route {route_id!r} differs from saved name {config.get('name')!r}"
            )
        results.append(
            {
                "route_id": route_id,
                "path": str(directory),
                "config": _artifact(config_path),
                "weights": _artifact(weights_path),
            }
        )
    return results


def train_router_checkpoint(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path | None,
    model_dir: str | Path,
    adapters: Mapping[str, str | Path],
    output_dir: str | Path,
    route_ids: Sequence[str] = DEFAULT_ROUTE_IDS,
    feature_contract: RouterFeatureContract = RouterFeatureContract(),
    training: RouterTrainingConfig = RouterTrainingConfig(),
    feature_batch_size: int = 64,
    cache_dir: str | Path | None = None,
    limit_per_route: int | None = None,
    progress: ProgressCallback | None = None,
) -> RouterTrainingResult:
    """Extract/cache features, fit the router, and save its exact local artifact contract."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("install molor[hf] to save a router checkpoint") from exc

    routes = tuple(route_ids)
    if len(routes) < 2 or routes[-1] != "base/null" or len(routes) != len(set(routes)):
        raise ValueError("route_ids must be unique, with base/null structurally last")
    train_source = Path(train_path).resolve()
    validation_source = Path(validation_path).resolve()
    test_source = None if test_path is None else Path(test_path).resolve()
    datasets = {
        "train": load_router_jsonl(train_source, route_ids=routes, limit_per_route=limit_per_route),
        "validation": load_router_jsonl(
            validation_source, route_ids=routes, limit_per_route=limit_per_route
        ),
    }
    if test_source is not None:
        datasets["test"] = load_router_jsonl(
            test_source, route_ids=routes, limit_per_route=limit_per_route
        )
    model_root = Path(model_dir).expanduser().resolve()
    config_path = model_root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        hidden_size = int(model_config["hidden_size"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"model config has no valid hidden_size: {config_path}") from exc
    embedding, embedding_metadata = load_input_embedding(
        model_root,
        tensor_name=feature_contract.embedding_tensor,
    )
    if embedding.shape[1] != hidden_size:
        raise ValueError("model hidden_size differs from the input embedding width")
    tokenizer = load_router_tokenizer(model_root)
    cache_root = None if cache_dir is None else Path(cache_dir).resolve()
    split_tensors: dict[str, tuple[Tensor, Tensor]] = {}
    cache_hits: dict[str, bool] = {}
    source_paths = {
        "train": train_source,
        "validation": validation_source,
        "test": test_source,
    }
    for split, examples in datasets.items():
        cache_path = None if cache_root is None else cache_root / f"{split}.safetensors"
        features, labels, hit = cached_prompt_features(
            source_paths[split],
            examples=examples,
            tokenizer=tokenizer,
            embedding=embedding,
            embedding_metadata=embedding_metadata,
            route_ids=routes,
            contract=feature_contract,
            cache_path=cache_path,
            batch_size=feature_batch_size,
            progress=(
                None if progress is None else lambda text, name=split: progress(f"{name}: {text}")
            ),
        )
        split_tensors[split] = (features, labels)
        cache_hits[split] = hit
    del embedding

    router, metrics = fit_router(
        *split_tensors["train"],
        *split_tensors["validation"],
        route_ids=routes,
        hidden_size=hidden_size,
        training=training,
        progress=progress,
    )
    if "test" in split_tensors:
        metrics["test"] = _split_metrics(
            router,
            *split_tensors["test"],
            route_ids=routes,
            batch_size=training.batch_size,
            device=torch.device("cpu"),
        )
    metrics["feature_cache_hits"] = cache_hits

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "router.safetensors"
    tensor_state = {
        f"network.{name}": tensor.detach().cpu().contiguous()
        for name, tensor in router.network.state_dict().items()
    }
    save_file(tensor_state, checkpoint_path)
    adapter_metadata = _adapter_artifacts(adapters, routes)
    dataset_metadata = {
        split: _artifact(path) for split, path in source_paths.items() if path is not None
    }
    sidecar = {
        "schema_version": ROUTER_CHECKPOINT_SCHEMA,
        "route_ids": list(routes),
        "route_to_index": {route_id: index for index, route_id in enumerate(routes)},
        "model": {
            "model_id": str(model_root),
            "architecture": next(iter(model_config.get("architectures") or ()), None),
            "hidden_size": hidden_size,
            "config": _artifact(config_path),
            "embedding": embedding_metadata,
            "tokenizer_artifacts": _tokenizer_artifacts(model_root),
        },
        "adapters": adapter_metadata,
        "feature_contract": feature_contract.to_dict(),
        "router_config": router.config.to_dict(),
        "training_config": training.to_dict(),
        "datasets": dataset_metadata,
        "metrics": metrics,
        "checkpoint": _artifact(checkpoint_path),
    }
    router_config_path = destination / "router_config.json"
    router_config_path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return RouterTrainingResult(
        output_dir=destination,
        checkpoint_path=checkpoint_path,
        config_path=router_config_path,
        metrics=metrics,
    )
