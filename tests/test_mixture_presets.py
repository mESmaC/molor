import hashlib
import json

import pytest

from molor.experimental.mixture_presets import (
    MIXTURE_COMPOSITION,
    MIXTURE_NORMALIZATION,
    MixtureLeafWeight,
    MixturePreset,
    MixturePresetRegistry,
    load_mixture_preset,
    mixture_preset_json_bytes,
)
from molor.identity import fingerprint_json


RACK = hashlib.sha256(b"rack-v1").hexdigest()
OTHER_RACK = hashlib.sha256(b"rack-v2").hexdigest()
LEAVES = ("formal-math-v1", "quantitative-math-v1", "science-reasoning-v1")


def _preset(
    preset_id: str = "math-balanced-v1",
    *,
    rack_fingerprint: str = RACK,
    leaves: tuple[MixtureLeafWeight, ...] | None = None,
) -> MixturePreset:
    return MixturePreset(
        preset_id=preset_id,
        revision="1",
        rack_fingerprint=rack_fingerprint,
        leaves=leaves
        or (
            MixtureLeafWeight("quantitative-math-v1", 0.65),
            MixtureLeafWeight("formal-math-v1", 0.35),
        ),
    )


def _write_preset(path, preset: MixturePreset) -> None:
    path.write_bytes(mixture_preset_json_bytes(preset))


def test_preset_has_deterministic_semantic_identity_and_canonical_leaf_order() -> None:
    first = _preset()
    second = _preset(
        leaves=(
            MixtureLeafWeight("formal-math-v1", 0.35),
            MixtureLeafWeight("quantitative-math-v1", 0.65),
        )
    )

    assert first == second
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint == fingerprint_json(first.content_dict())
    assert first.to_dict()["content_sha256"] == first.fingerprint
    assert tuple(first.route) == ("formal-math-v1", "quantitative-math-v1")
    assert first.normalization == MIXTURE_NORMALIZATION
    assert first.composition == MIXTURE_COMPOSITION
    with pytest.raises(TypeError):
        first.route["formal-math-v1"] = 1.0


def test_preset_rejects_ambiguous_or_unsafe_route_semantics() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _preset(leaves=(MixtureLeafWeight("formal-math-v1", 0.9),))
    with pytest.raises(ValueError, match="repeat"):
        _preset(
            leaves=(
                MixtureLeafWeight("formal-math-v1", 0.5),
                MixtureLeafWeight("formal-math-v1", 0.5),
            )
        )
    with pytest.raises(ValueError, match="adapter leaf"):
        MixtureLeafWeight("base/null", 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        MixtureLeafWeight("formal-math-v1", float("nan"))
    with pytest.raises(TypeError, match="real number"):
        MixtureLeafWeight("formal-math-v1", True)
    with pytest.raises(ValueError, match="normalization"):
        MixturePreset(
            "math-balanced-v1",
            "1",
            RACK,
            (MixtureLeafWeight("formal-math-v1", 1.0),),
            normalization="runtime_normalizes",
        )
    with pytest.raises(ValueError, match="composition"):
        MixturePreset(
            "math-balanced-v1",
            "1",
            RACK,
            (MixtureLeafWeight("formal-math-v1", 1.0),),
            composition="merge_weights",
        )


def test_strict_loader_round_trips_and_rejects_semantic_tampering(tmp_path) -> None:
    preset = _preset()
    path = tmp_path / f"{preset.preset_id}.json"
    _write_preset(path, preset)

    loaded = load_mixture_preset(path)
    assert loaded == preset
    assert mixture_preset_json_bytes(loaded) == path.read_bytes()

    payload = preset.to_dict()
    payload["leaves"][0]["weight"] = 0.34
    payload["leaves"][1]["weight"] = 0.66
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content_sha256"):
        load_mixture_preset(path)


def test_strict_loader_rejects_unknown_fields_duplicate_keys_and_nonstandard_numbers(
    tmp_path,
) -> None:
    preset = _preset()
    payload = preset.to_dict()
    payload["unexpected"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_mixture_preset(unknown)

    duplicate = tmp_path / "duplicate.json"
    raw = mixture_preset_json_bytes(preset).decode("utf-8").rstrip()
    duplicate.write_text(raw[:-1] + ', "preset_id": "duplicate"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_mixture_preset(duplicate)

    nonstandard = tmp_path / "nan.json"
    nonstandard.write_text(raw.replace("0.35", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported numeric constant"):
        load_mixture_preset(nonstandard)


def test_loader_requires_canonical_leaf_order_and_lowercase_digest(tmp_path) -> None:
    preset = _preset()
    payload = preset.to_dict()
    payload["leaves"] = list(reversed(payload["leaves"]))
    unordered = tmp_path / "unordered.json"
    unordered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered canonically"):
        load_mixture_preset(unordered)

    payload = preset.to_dict()
    payload["content_sha256"] = preset.fingerprint.upper()
    uppercase = tmp_path / "uppercase.json"
    uppercase.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase"):
        load_mixture_preset(uppercase)


def test_registry_binds_every_preset_to_one_exact_rack_and_leaf_inventory() -> None:
    math = _preset()
    science = _preset(
        "science-only-v1",
        leaves=(MixtureLeafWeight("science-reasoning-v1", 1.0),),
    )
    registry = MixturePresetRegistry(RACK, LEAVES, (science, math))

    assert tuple(preset.preset_id for preset in registry.presets) == (
        "math-balanced-v1",
        "science-only-v1",
    )
    assert registry.resolve("math-balanced-v1") is math
    assert registry.resolve("math-balanced-v1", expected_fingerprint=math.fingerprint) is math
    assert registry.fingerprint == fingerprint_json(registry.content_dict())
    with pytest.raises(KeyError, match="unknown mixture preset"):
        registry.resolve("missing-v1")
    with pytest.raises(ValueError, match="expected fingerprint"):
        registry.resolve("math-balanced-v1", expected_fingerprint="0" * 64)

    with pytest.raises(ValueError, match="different expert rack"):
        MixturePresetRegistry(RACK, LEAVES, (_preset(rack_fingerprint=OTHER_RACK),))
    with pytest.raises(ValueError, match="unknown rack leaves"):
        MixturePresetRegistry(
            RACK,
            LEAVES,
            (
                _preset(
                    leaves=(MixtureLeafWeight("commonsense-logic-v1", 1.0),),
                ),
            ),
        )


def test_registry_directory_load_is_closed_content_addressed_and_filename_bound(tmp_path) -> None:
    math = _preset()
    science = _preset(
        "science-only-v1",
        leaves=(MixtureLeafWeight("science-reasoning-v1", 1.0),),
    )
    _write_preset(tmp_path / f"{math.preset_id}.json", math)
    _write_preset(tmp_path / f"{science.preset_id}.json", science)

    first = MixturePresetRegistry.load_directory(
        tmp_path,
        rack_fingerprint=RACK,
        leaf_ids=LEAVES,
        expected_presets={
            math.preset_id: math.fingerprint,
            science.preset_id: science.fingerprint,
        },
    )
    second = MixturePresetRegistry.load_directory(
        tmp_path,
        rack_fingerprint=RACK,
        leaf_ids=LEAVES,
        expected_registry_fingerprint=first.fingerprint,
    )
    assert second == first

    with pytest.raises(ValueError, match="expected registry set"):
        MixturePresetRegistry.load_directory(
            tmp_path,
            rack_fingerprint=RACK,
            leaf_ids=LEAVES,
            expected_presets={math.preset_id: math.fingerprint},
        )
    with pytest.raises(ValueError, match="registry fingerprint mismatch"):
        MixturePresetRegistry.load_directory(
            tmp_path,
            rack_fingerprint=RACK,
            leaf_ids=LEAVES,
            expected_registry_fingerprint="0" * 64,
        )


def test_registry_directory_rejects_unexpected_entries_and_misnamed_presets(tmp_path) -> None:
    preset = _preset()
    _write_preset(tmp_path / "wrong-name.json", preset)
    with pytest.raises(ValueError, match="filename"):
        MixturePresetRegistry.load_directory(
            tmp_path,
            rack_fingerprint=RACK,
            leaf_ids=LEAVES,
        )

    (tmp_path / "wrong-name.json").unlink()
    _write_preset(tmp_path / f"{preset.preset_id}.json", preset)
    (tmp_path / "README.txt").write_text("not part of the registry", encoding="utf-8")
    with pytest.raises(ValueError, match="only regular lowercase .json"):
        MixturePresetRegistry.load_directory(
            tmp_path,
            rack_fingerprint=RACK,
            leaf_ids=LEAVES,
        )
