"""Content-addressed model/adapter/router identity for benchmark provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path

from molor.benchmark.models import fingerprint


LOCK_SCHEMA = "molor.benchmark-stack-lock.v2"
_HASH_BUFFER_SIZE = 8 * 1024 * 1024
_BASE_METADATA = {
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
}
_TOKENIZER_PAYLOADS = {
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "vocab.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
}
_WEIGHT_INDEXES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
_BASE_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


class _IncompleteArtifact(ValueError):
    """A configured artifact directory cannot be locked completely."""


def _file_sha256(path: Path) -> str:
    """Hash the bytes on every lock operation.

    A size/mtime cache is intentionally not used: both attributes can be restored after
    an in-place mutation, which would make a benchmark silently retain a stale digest.
    Locking happens once per run, so a full read is the appropriate correctness tradeoff.
    """

    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_BUFFER_SIZE), b""):
            digest.update(chunk)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise OSError(f"artifact changed while it was being hashed: {path}")
    return digest.hexdigest()


def _indexed_weight_files(directory: Path, index_path: Path) -> set[Path]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _IncompleteArtifact(f"invalid weight index {index_path.name}: {exc}") from exc
    weight_map = payload.get("weight_map") if isinstance(payload, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise _IncompleteArtifact(f"weight index has no non-empty weight_map: {index_path.name}")

    files: set[Path] = set()
    for raw_name in weight_map.values():
        if not isinstance(raw_name, str) or not raw_name:
            raise _IncompleteArtifact(
                f"weight index contains an invalid shard name: {index_path.name}"
            )
        relative = Path(raw_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise _IncompleteArtifact(
                f"weight index references a path outside the checkpoint: {raw_name}"
            )
        candidate = directory / relative
        if not candidate.is_file():
            raise _IncompleteArtifact(f"weight index references a missing shard: {raw_name}")
        files.add(candidate)
    return files


def _base_files(directory: Path) -> list[Path]:
    config = directory / "config.json"
    if not config.is_file():
        raise _IncompleteArtifact("base checkpoint is missing config.json")

    tokenizer_payloads = [directory / name for name in _TOKENIZER_PAYLOADS]
    tokenizer_payloads = [path for path in tokenizer_payloads if path.is_file()]
    if not tokenizer_payloads:
        expected = ", ".join(sorted(_TOKENIZER_PAYLOADS))
        raise _IncompleteArtifact(f"base checkpoint has no tokenizer payload ({expected})")

    metadata = {
        directory / name
        for name in _BASE_METADATA | _WEIGHT_INDEXES
        if (directory / name).is_file()
    }
    weights = {
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() in _BASE_WEIGHT_SUFFIXES
    }
    for index_name in _WEIGHT_INDEXES:
        index_path = directory / index_name
        if index_path.is_file():
            weights.update(_indexed_weight_files(directory, index_path))
    if not weights:
        raise _IncompleteArtifact("base checkpoint has no model weight file")
    return sorted(metadata | weights, key=lambda path: path.name.casefold())


def _expert_files(directory: Path) -> list[Path]:
    config = directory / "adapter_config.json"
    weights = [
        directory / "adapter_model.safetensors",
        directory / "adapter_model.bin",
    ]
    if not config.is_file():
        raise _IncompleteArtifact("expert is missing adapter_config.json")
    weights = [path for path in weights if path.is_file()]
    if not weights:
        raise _IncompleteArtifact(
            "expert is missing adapter_model.safetensors or adapter_model.bin"
        )
    return [config, *weights]


def _router_files(directory: Path) -> list[Path]:
    config = directory / "router_config.json"
    weights = [directory / "router.safetensors", directory / "router.bin"]
    if not config.is_file():
        raise _IncompleteArtifact("router is missing router_config.json")
    weights = [path for path in weights if path.is_file()]
    if not weights:
        raise _IncompleteArtifact("router is missing router.safetensors or router.bin")
    return [config, *weights]


def _artifact_files(directory: Path, role: str) -> list[Path]:
    if role == "base":
        return _base_files(directory)
    if role == "expert":
        return _expert_files(directory)
    if role == "router":
        return _router_files(directory)
    raise ValueError(f"unsupported artifact role {role!r}")


def _lock_directory(directory: Path, role: str) -> dict[str, object]:
    resolved = directory.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{role} directory does not exist: {resolved}")
    files: list[dict[str, object]] = []
    for path in _artifact_files(resolved, role):
        stat = path.stat()
        print(f"Fingerprinting {role}: {path.name} ({stat.st_size / 2**20:,.1f} MiB)", flush=True)
        files.append(
            {
                "name": path.relative_to(resolved).as_posix(),
                "size_bytes": stat.st_size,
                "sha256": _file_sha256(path),
            }
        )
    content_fingerprint = fingerprint({"role": role, "files": files})
    return {
        "role": role,
        "path": str(resolved),
        "content_fingerprint": content_fingerprint,
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
        "complete": True,
    }


def _lock_file(path: Path, role: str) -> dict[str, object]:
    """Lock a self-contained model artifact such as a GGUF checkpoint."""

    resolved = path.expanduser().resolve()
    if role != "base" or resolved.suffix.casefold() != ".gguf":
        raise _IncompleteArtifact(
            f"single-file locking only supports a .gguf base checkpoint, not {resolved.name}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"{role} artifact does not exist: {resolved}")
    stat = resolved.stat()
    print(
        f"Fingerprinting {role}: {resolved.name} ({stat.st_size / 2**20:,.1f} MiB)",
        flush=True,
    )
    files = [
        {
            "name": resolved.name,
            "size_bytes": stat.st_size,
            "sha256": _file_sha256(resolved),
        }
    ]
    return {
        "role": role,
        "path": str(resolved),
        "content_fingerprint": fingerprint({"role": role, "files": files}),
        "size_bytes": stat.st_size,
        "files": files,
        "complete": True,
    }


def _unresolved(role: str, identifier: object, reason: str) -> dict[str, str]:
    rendered = str(identifier) if identifier not in (None, "") else "<missing>"
    return {"role": role, "identifier": rendered, "reason": reason}


def _try_lock(
    artifacts: list[dict[str, object]],
    unresolved: list[dict[str, str]],
    *,
    role: str,
    identifier: object,
) -> None:
    if not isinstance(identifier, str) or not identifier:
        unresolved.append(_unresolved(role, identifier, "no artifact path was provided"))
        return
    path = Path(identifier).expanduser()
    if not path.is_dir() and not path.is_file():
        unresolved.append(_unresolved(role, identifier, "artifact path is unavailable"))
        return
    try:
        if path.is_file():
            artifacts.append(_lock_file(path, role))
        else:
            artifacts.append(_lock_directory(path, role))
    except (FileNotFoundError, OSError, _IncompleteArtifact) as exc:
        unresolved.append(_unresolved(role, identifier, str(exc)))


def lock_backend_identity(
    identity: Mapping[str, object],
    *,
    cache_path: Path,
) -> dict[str, object]:
    """Resolve every configured backend role to complete local bytes when possible.

    ``cache_path`` remains part of the public call contract, but digest caching is disabled
    deliberately. Rehashing once per benchmark run prevents a same-size/same-mtime mutation
    from inheriting an old SHA-256 value.
    """

    del cache_path
    artifacts: list[dict[str, object]] = []
    unresolved: list[dict[str, str]] = []
    required_roles: list[dict[str, object]] = []

    model_id = identity.get("model_id")
    model_artifact = identity.get("model_artifact_path")
    base_identifier = model_artifact if model_artifact not in (None, "") else model_id
    required_roles.append({"role": "base", "identifier": base_identifier})
    _try_lock(artifacts, unresolved, role="base", identifier=base_identifier)

    raw_experts = identity.get("configured_expert_paths")
    expert_paths: list[object] = []
    if isinstance(raw_experts, Sequence) and not isinstance(raw_experts, (str, bytes)):
        expert_paths = list(raw_experts)
    elif raw_experts not in (None, []):
        unresolved.append(
            _unresolved("expert", raw_experts, "configured_expert_paths is not a sequence")
        )
    for index, raw_path in enumerate(expert_paths):
        required_roles.append({"role": "expert", "index": index, "identifier": raw_path})
        _try_lock(artifacts, unresolved, role="expert", identifier=raw_path)

    advertised_experts = identity.get("experts")
    if isinstance(advertised_experts, Sequence) and not isinstance(
        advertised_experts, (str, bytes)
    ):
        for index in range(len(expert_paths), len(advertised_experts)):
            name = advertised_experts[index]
            required_roles.append({"role": "expert", "index": index, "identifier": name})
            unresolved.append(
                _unresolved("expert", name, "advertised expert has no configured artifact path")
            )

    router = identity.get("router_checkpoint")
    if router not in (None, "") or identity.get("router_loaded") is True:
        required_roles.append({"role": "router", "identifier": router})
        _try_lock(artifacts, unresolved, role="router", identifier=router)

    stable_identity = dict(identity)
    stable_identity.pop("model_artifact_path", None)
    stable_identity.pop("configured_expert_paths", None)
    stable_identity.pop("router_checkpoint", None)
    if identity.get("backend") != "openai-compatible" and any(
        artifact.get("role") == "base" for artifact in artifacts
    ):
        # Exact model/tokenizer bytes replace a machine-specific local directory name.
        stable_identity.pop("model_id", None)
    # Endpoint is retained. In particular, OpenAI-compatible endpoints and model aliases
    # are behavior-affecting identities even if they happen to advertise the same label.

    fully_content_addressed = (
        len(artifacts) == len(required_roles)
        and bool(required_roles)
        and not unresolved
        and all(artifact.get("complete") is True for artifact in artifacts)
    )
    payload = {
        "schema_version": LOCK_SCHEMA,
        "hash_policy": "sha256-full-read-every-lock",
        "backend_identity": dict(identity),
        "stable_backend_identity": stable_identity,
        "required_roles": required_roles,
        "artifacts": artifacts,
        "unresolved": unresolved,
        "fully_content_addressed": fully_content_addressed,
    }
    payload["stack_fingerprint"] = fingerprint(
        {
            "schema_version": LOCK_SCHEMA,
            "stable_backend_identity": stable_identity,
            "artifact_fingerprints": [
                {"role": item["role"], "content_fingerprint": item["content_fingerprint"]}
                for item in artifacts
            ],
            "unresolved": unresolved,
        }
    )
    return payload


__all__ = ["LOCK_SCHEMA", "lock_backend_identity"]
