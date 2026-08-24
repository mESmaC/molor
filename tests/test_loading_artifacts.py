from dataclasses import replace
import json
from pathlib import Path

import pytest

from molor.loading import (
    ArtifactVerificationError,
    LocalArtifactMaterializer,
    WeightIndexValidationError,
)
from molor.manifests import (
    JSON_MEDIA_TYPE,
    SAFETENSORS_MEDIA_TYPE,
    ArtifactDescriptor,
    BaseModelManifest,
)


def write_artifact(
    root: Path,
    name: str,
    content: bytes,
    media_type: str,
) -> ArtifactDescriptor:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return ArtifactDescriptor.from_file(path, logical_name=name, media_type=media_type)


def base_manifest(
    root: Path,
    *,
    index_content: bytes | None = None,
    quantization: str = "none",
    custom_code: tuple[ArtifactDescriptor, ...] = (),
) -> BaseModelManifest:
    config = write_artifact(root, "config.json", b"{}", JSON_MEDIA_TYPE)
    tokenizer = write_artifact(root, "tokenizer.json", b"{}", JSON_MEDIA_TYPE)
    special_tokens = write_artifact(
        root,
        "special_tokens_map.json",
        b"{}",
        JSON_MEDIA_TYPE,
    )
    chat_template = write_artifact(
        root,
        "chat_template.jinja",
        b"{{ messages }}",
        "text/plain",
    )
    if index_content is None:
        weights = (
            write_artifact(
                root,
                "model.safetensors",
                b"safe-single-shard",
                SAFETENSORS_MEDIA_TYPE,
            ),
        )
        weight_index = None
    else:
        weights = (
            write_artifact(
                root,
                "model-00001-of-00002.safetensors",
                b"safe-shard-one",
                SAFETENSORS_MEDIA_TYPE,
            ),
            write_artifact(
                root,
                "model-00002-of-00002.safetensors",
                b"safe-shard-two",
                SAFETENSORS_MEDIA_TYPE,
            ),
        )
        weight_index = write_artifact(
            root,
            "model.safetensors.index.json",
            index_content,
            JSON_MEDIA_TYPE,
        )
    return BaseModelManifest(
        model_id="local/TestModel",
        requested_revision="main",
        resolved_revision="0123456789abcdef0123456789abcdef01234567",
        architecture="FakeForCausalLM",
        dtype="torch.BFLOAT16",
        quantization=quantization,
        config=config,
        weights=weights,
        tokenizer=(tokenizer,),
        special_tokens=special_tokens,
        weight_index=weight_index,
        chat_template=chat_template,
        custom_code=custom_code,
    )


def valid_index() -> bytes:
    return json.dumps(
        {
            "metadata": {"total_size": 24},
            "weight_map": {
                "model.embed.weight": "model-00001-of-00002.safetensors",
                "model.output.weight": "model-00002-of-00002.safetensors",
            },
        },
        separators=(",", ":"),
    ).encode()


def test_materializer_copies_only_declared_bytes_and_has_path_independent_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = base_manifest(source, index_content=valid_index())
    (source / "rogue.bin").write_bytes(b"pickle-shaped-but-untrusted")
    (source / "modeling_remote.py").write_text("raise RuntimeError", encoding="utf-8")
    parent = tmp_path / "staging"
    parent.mkdir()
    materializer = LocalArtifactMaterializer(chunk_size=3)

    first = materializer.materialize(manifest, source, staging_parent=parent)
    second = materializer.materialize(manifest, source, staging_parent=parent)
    first_root = first.root
    second_root = second.root
    try:
        staged_files = {
            path.relative_to(first.root).as_posix()
            for path in first.root.rglob("*")
            if path.is_file()
        }
        assert "rogue.bin" not in staged_files
        assert "modeling_remote.py" not in staged_files
        assert staged_files == {
            artifact.descriptor.logical_name for artifact in first.evidence.artifacts
        }
        assert first.root != second.root
        assert first.evidence.fingerprint == second.evidence.fingerprint
        assert first.evidence.base_fingerprint == manifest.fingerprint
        assert manifest.dtype == "bfloat16"
    finally:
        first.close()
        second.close()
    assert not first_root.exists()
    assert not second_root.exists()


def test_materializer_rejects_size_and_digest_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = base_manifest(source)
    (source / "model.safetensors").write_bytes(b"changed-after-manifest")

    with pytest.raises(ArtifactVerificationError, match="size differs"):
        LocalArtifactMaterializer(chunk_size=2).materialize(
            manifest,
            source,
            staging_parent=tmp_path,
        )

    original = base_manifest(source)
    bad_config = replace(original.config, sha256="f" * 64)
    digest_mismatch = replace(original, config=bad_config)
    with pytest.raises(ArtifactVerificationError, match="SHA-256 differs"):
        LocalArtifactMaterializer().materialize(
            digest_mismatch,
            source,
            staging_parent=tmp_path,
        )


@pytest.mark.parametrize(
    ("index_content", "message"),
    [
        (
            b'{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors"},'
            b'"weight_map":{"b":"model-00002-of-00002.safetensors"}}',
            "duplicate JSON key",
        ),
        (
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "undeclared.safetensors",
                    },
                }
            ).encode(),
            "undeclared shards",
        ),
        (
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {"a": "model-00001-of-00002.safetensors"},
                }
            ).encode(),
            "omits declared shards",
        ),
        (
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "../model-00002-of-00002.safetensors",
                    },
                }
            ).encode(),
            "unsafe shard",
        ),
        (
            json.dumps(
                {
                    "metadata": {},
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors",
                    },
                    "extra": {},
                }
            ).encode(),
            "exactly metadata and weight_map",
        ),
    ],
)
def test_materializer_strictly_validates_weight_index(
    tmp_path: Path,
    index_content: bytes,
    message: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = base_manifest(source, index_content=index_content)

    with pytest.raises(WeightIndexValidationError, match=message):
        LocalArtifactMaterializer().materialize(
            manifest,
            source,
            staging_parent=tmp_path,
        )


def test_base_manifest_requires_index_for_shards_and_valid_media_types(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = base_manifest(source, index_content=valid_index())

    with pytest.raises(ValueError, match="explicit weight_index"):
        replace(manifest, weight_index=None)
    with pytest.raises(ValueError, match="JSON media type"):
        replace(
            manifest,
            weight_index=replace(
                manifest.weight_index,
                media_type="application/octet-stream",
            ),
        )
    with pytest.raises(ValueError, match="dtype"):
        replace(manifest, dtype="float8")
