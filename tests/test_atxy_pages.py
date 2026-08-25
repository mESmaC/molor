import pytest

from molor.experimental.atxy_pages import (
    ATXY_ADDRESS_COUNT,
    ATXY_PAGES_PER_LAYER,
    AtxyPageAddress,
    token_capacity_per_layer,
)


def test_layer_page_mapping_round_trips_at_boundaries() -> None:
    first = AtxyPageAddress.for_layer_page(0, 0)
    last_qwen_layer_page = AtxyPageAddress.for_layer_page(27, ATXY_PAGES_PER_LAYER - 1)

    assert first.text == "0.000"
    assert first.packed == 0
    assert last_qwen_layer_page.text == "R.ZZZ"
    assert last_qwen_layer_page.layer_index == 27
    assert last_qwen_layer_page.page_id == 46_655
    assert AtxyPageAddress.from_packed(last_qwen_layer_page.packed) == last_qwen_layer_page
    assert AtxyPageAddress.from_text("r.zzz") == last_qwen_layer_page


def test_atxy_page_geometry_covers_ten_million_tokens_at_256_tokens_per_page() -> None:
    token_count = 10_000_000
    final_page_id = (token_count - 1) // 256
    final_address = AtxyPageAddress.for_layer_page(27, final_page_id)

    assert final_page_id == 39_062
    assert final_address.page_id == final_page_id
    assert token_capacity_per_layer(256) == 11_943_936
    assert 28 * (final_page_id + 1) < ATXY_ADDRESS_COUNT


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: AtxyPageAddress.for_layer_page(36, 0), "layer_index"),
        (lambda: AtxyPageAddress.for_layer_page(0, ATXY_PAGES_PER_LAYER), "page_id"),
        (lambda: AtxyPageAddress.from_packed(ATXY_ADDRESS_COUNT), "packed"),
        (lambda: AtxyPageAddress.from_text("A-XYZ"), "A.TXY"),
        (lambda: token_capacity_per_layer(0), "positive"),
    ],
)
def test_invalid_addresses_fail_closed(call, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        call()
