import torch

from molor.experimental.atxy_vq_lab import (
    _decode_grouped,
    _encode_grouped,
    _fit_grouped_codecs,
    _progressive_storage,
)


def _payload(seed: int, *, layers: int = 2, tokens: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(layers, tokens, 2, 4, generator=generator)


def test_global_and_layer_groupings_round_trip_expected_shapes() -> None:
    calibration = (_payload(1), _payload(2))
    heldout = _payload(3)

    for grouping, codec_count in (("global", 1), ("layer", 2), ("layer-head", 4)):
        codecs = _fit_grouped_codecs(
            calibration,
            grouping=grouping,
            seed=9,
            max_iterations=1,
            codebook_dtype=torch.bfloat16,
            tuple_count=2,
        )
        fields, sample_counts = _encode_grouped(heldout, codecs, grouping=grouping)
        reconstructed = _decode_grouped(
            fields,
            codecs,
            grouping=grouping,
            shape=heldout.shape,
            tuple_depth=2,
        )

        assert len(codecs) == len(fields) == codec_count
        assert sum(sample_counts) == heldout.shape[0] * heldout.shape[1] * heldout.shape[2]
        assert reconstructed.shape == heldout.shape
        assert reconstructed.dtype == torch.float32


def test_progressive_storage_charges_only_materialized_fields_and_codebooks() -> None:
    calibration = (_payload(10, layers=1),)
    heldout = _payload(11, layers=1)
    codecs = _fit_grouped_codecs(
        calibration,
        grouping="global",
        seed=7,
        max_iterations=1,
        codebook_dtype=torch.bfloat16,
        tuple_count=2,
    )
    fields, _ = _encode_grouped(heldout, codecs, grouping="global")
    native_bytes = heldout.numel() * heldout.element_size()

    first = _progressive_storage(fields, codecs, tuple_depth=1, native_bytes=native_bytes)
    full = _progressive_storage(fields, codecs, tuple_depth=2, native_bytes=native_bytes)

    assert first["bits_per_vector"] == 32
    assert full["bits_per_vector"] == 64
    assert full["codebook_bytes"] == codecs[0].codebook_bytes
    assert full["field_bytes"] == fields[0].numel() * fields[0].element_size()
    assert full["total_bytes"] == 2 * first["total_bytes"]
