from __future__ import annotations

import json
from pathlib import Path

import pytest

from molor.identity import canonical_json_bytes
from molor.loading import (
    ManifestSerializationError,
    canonicalize_base_model_manifest_json,
    dumps_base_model_manifest,
    load_base_model_manifest,
    loads_base_model_manifest,
)
from molor.manifests import (
    JSON_MEDIA_TYPE,
    SAFETENSORS_MEDIA_TYPE,
    ArtifactDescriptor,
    BaseModelManifest,
)


def artifact(
    logical_name: str,
    marker: str,
    *,
    media_type: str,
    size_bytes: int = 100,
) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        logical_name=logical_name,
        media_type=media_type,
        sha256=marker * 64,
        size_bytes=size_bytes,
    )


def manifest() -> BaseModelManifest:
    return BaseModelManifest(
        model_id="local/StrictModel",
        requested_revision="main",
        resolved_revision="0123456789abcdef0123456789abcdef01234567",
        architecture="StrictForCausalLM",
        dtype="bfloat16",
        quantization="none",
        config=artifact("config.json", "a", media_type=JSON_MEDIA_TYPE),
        weights=(
            artifact(
                "model-00001-of-00002.safetensors",
                "b",
                media_type=SAFETENSORS_MEDIA_TYPE,
            ),
            artifact(
                "model-00002-of-00002.safetensors",
                "c",
                media_type=SAFETENSORS_MEDIA_TYPE,
            ),
        ),
        tokenizer=(
            artifact("tokenizer.json", "d", media_type=JSON_MEDIA_TYPE),
            artifact("tokenizer_config.json", "e", media_type=JSON_MEDIA_TYPE),
        ),
        special_tokens=artifact(
            "special_tokens_map.json",
            "f",
            media_type=JSON_MEDIA_TYPE,
        ),
        weight_index=artifact(
            "model.safetensors.index.json",
            "1",
            media_type=JSON_MEDIA_TYPE,
        ),
        chat_template=artifact(
            "chat_template.jinja",
            "2",
            media_type="text/plain",
        ),
        custom_code=(),
    )


def payload() -> dict[str, object]:
    return manifest().to_dict()


def encoded(value: object, *, allow_nan: bool = False) -> bytes:
    return json.dumps(value, allow_nan=allow_nan).encode("utf-8")


def test_canonical_dump_round_trip_and_file_load(tmp_path: Path) -> None:
    original = manifest()
    canonical = dumps_base_model_manifest(original)

    assert canonical == canonical_json_bytes(original.to_dict())
    assert loads_base_model_manifest(canonical) == original
    assert loads_base_model_manifest(canonical.decode("utf-8")).fingerprint == original.fingerprint

    noncanonical = json.dumps(original.to_dict(), indent=2, sort_keys=False).encode()
    assert canonicalize_base_model_manifest_json(noncanonical) == canonical

    path = tmp_path / "base-manifest.json"
    path.write_bytes(canonical)
    assert load_base_model_manifest(path) == original


@pytest.mark.parametrize("schema_version", ["molor.base-model.v1", "molor.base-model.v3", "x"])
def test_rejects_legacy_and_unknown_schema_versions(schema_version: str) -> None:
    data = payload()
    data["schema_version"] = schema_version

    with pytest.raises(ManifestSerializationError, match="unsupported.*schema_version"):
        loads_base_model_manifest(encoded(data))


def test_rejects_duplicate_keys_at_top_and_nested_levels() -> None:
    canonical = dumps_base_model_manifest(manifest()).decode()
    duplicate_top = canonical.replace(
        '"model_id":"local/StrictModel"',
        '"model_id":"local/StrictModel","model_id":"other/Model"',
    )
    duplicate_nested = canonical.replace(
        '"logical_name":"config.json"',
        '"logical_name":"config.json","logical_name":"other.json"',
        1,
    )

    with pytest.raises(ManifestSerializationError, match="duplicate key 'model_id'"):
        loads_base_model_manifest(duplicate_top)
    with pytest.raises(ManifestSerializationError, match="duplicate key 'logical_name'"):
        loads_base_model_manifest(duplicate_nested)


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_rejects_non_exact_base_field_sets(mutation: str) -> None:
    data = payload()
    if mutation == "extra":
        data["untrusted"] = "value"
    else:
        del data["architecture"]

    with pytest.raises(ManifestSerializationError, match="invalid field set"):
        loads_base_model_manifest(encoded(data))


@pytest.mark.parametrize("mutation", ["extra", "missing", "not-object"])
def test_rejects_malformed_nested_artifact_records(mutation: str) -> None:
    data = payload()
    config = data["config"]
    assert isinstance(config, dict)
    if mutation == "extra":
        config["schema_version"] = "molor.artifact.v1"
    elif mutation == "missing":
        del config["sha256"]
    else:
        data["config"] = "config.json"

    with pytest.raises(ManifestSerializationError, match="base manifest.config"):
        loads_base_model_manifest(encoded(data))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", 12, "model_id must be a string"),
        ("weights", {}, "weights must be a JSON array"),
        ("tokenizer", "tokenizer.json", "tokenizer must be a JSON array"),
        ("weight_index", False, "weight_index must be a JSON object"),
        ("chat_template", [], "chat_template must be a JSON object"),
        ("custom_code", {}, "custom_code must be a JSON array"),
    ],
)
def test_rejects_wrong_base_field_types(field: str, value: object, message: str) -> None:
    data = payload()
    data[field] = value

    with pytest.raises(ManifestSerializationError, match=message):
        loads_base_model_manifest(encoded(data))


@pytest.mark.parametrize("invalid_size", [True, False, -1, 1.5, "100", None])
def test_rejects_boolean_and_non_integer_artifact_sizes(invalid_size: object) -> None:
    data = payload()
    config = data["config"]
    assert isinstance(config, dict)
    config["size_bytes"] = invalid_size

    with pytest.raises(ManifestSerializationError, match="non-negative integer"):
        loads_base_model_manifest(encoded(data))


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_rejects_non_finite_json_numbers(number: str) -> None:
    canonical = dumps_base_model_manifest(manifest()).decode()
    mutated = canonical.replace('"size_bytes":100', f'"size_bytes":{number}', 1)

    with pytest.raises(ManifestSerializationError, match="non-finite"):
        loads_base_model_manifest(mutated)


def test_rejects_invalid_nested_values_through_typed_constructors() -> None:
    data = payload()
    config = data["config"]
    assert isinstance(config, dict)
    config["logical_name"] = "../config.json"

    with pytest.raises(ManifestSerializationError, match="valid artifact descriptor"):
        loads_base_model_manifest(encoded(data))

    data = payload()
    data["dtype"] = "float8"
    with pytest.raises(ManifestSerializationError, match="typed contract"):
        loads_base_model_manifest(encoded(data))


@pytest.mark.parametrize("root", [[], None, "manifest", 7, True])
def test_rejects_non_object_roots(root: object) -> None:
    with pytest.raises(ManifestSerializationError, match="must be a JSON object"):
        loads_base_model_manifest(encoded(root))


def test_rejects_invalid_utf8_malformed_json_and_wrong_input_type() -> None:
    with pytest.raises(ManifestSerializationError, match="strict UTF-8"):
        loads_base_model_manifest(b"\xff\xfe")
    with pytest.raises(ManifestSerializationError, match="valid strict JSON"):
        loads_base_model_manifest(b'{"schema_version":')
    with pytest.raises(TypeError, match="str or bytes"):
        loads_base_model_manifest(bytearray(b"{}"))  # type: ignore[arg-type]


def test_input_and_output_byte_limits_are_enforced(tmp_path: Path) -> None:
    original = manifest()
    canonical = dumps_base_model_manifest(original)
    too_small = len(canonical) - 1

    with pytest.raises(ManifestSerializationError, match="byte limit"):
        loads_base_model_manifest(canonical, maximum_bytes=too_small)
    with pytest.raises(ManifestSerializationError, match="byte limit"):
        dumps_base_model_manifest(original, maximum_bytes=too_small)

    path = tmp_path / "oversize.json"
    path.write_bytes(canonical)
    with pytest.raises(ManifestSerializationError, match="byte limit"):
        load_base_model_manifest(path, maximum_bytes=too_small)

    with pytest.raises(ValueError, match="positive integer"):
        loads_base_model_manifest(canonical, maximum_bytes=True)  # type: ignore[arg-type]
