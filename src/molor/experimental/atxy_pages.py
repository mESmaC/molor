"""Exact ATXY addresses for layer-sharded KV pages.

This module deliberately implements only physical identity.  ATXY page
addresses are neither semantic routing hashes nor compressed KV payloads.
"""

from __future__ import annotations

from dataclasses import dataclass


ATXY_RADIX = 36
ATXY_FIELD_COUNT = 4
ATXY_ADDRESS_COUNT = ATXY_RADIX**ATXY_FIELD_COUNT
ATXY_PAGES_PER_LAYER = ATXY_RADIX**3
_DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _field(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < ATXY_RADIX:
        raise ValueError(f"{name} must be an integer from 0 through {ATXY_RADIX - 1}")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class AtxyPageAddress:
    """One immutable ``A.TXY`` address.

    For the per-layer KV layout, ``A`` is the layer and ``TXY`` is the
    three-digit base-36 page number.  Model, sequence, route, RoPE convention,
    and codec version remain part of the surrounding store namespace.
    """

    a: int
    t: int
    x: int
    y: int

    def __post_init__(self) -> None:
        for name in ("a", "t", "x", "y"):
            _field(getattr(self, name), name)

    @classmethod
    def for_layer_page(cls, layer_index: int, page_id: int) -> AtxyPageAddress:
        """Map a layer and page ID to the literal ATXY page namespace."""

        layer_index = _field(layer_index, "layer_index")
        page_id = _non_negative_int(page_id, "page_id")
        if page_id >= ATXY_PAGES_PER_LAYER:
            raise ValueError(
                f"page_id must be below {ATXY_PAGES_PER_LAYER} for a three-field ATXY page"
            )
        t, remainder = divmod(page_id, ATXY_RADIX**2)
        x, y = divmod(remainder, ATXY_RADIX)
        return cls(layer_index, t, x, y)

    @classmethod
    def from_packed(cls, packed: int) -> AtxyPageAddress:
        """Decode a packed integer in the canonical ``A,T,X,Y`` order."""

        packed = _non_negative_int(packed, "packed")
        if packed >= ATXY_ADDRESS_COUNT:
            raise ValueError(f"packed must be below {ATXY_ADDRESS_COUNT}")
        a_t_x, y = divmod(packed, ATXY_RADIX)
        a_t, x = divmod(a_t_x, ATXY_RADIX)
        a, t = divmod(a_t, ATXY_RADIX)
        return cls(a, t, x, y)

    @classmethod
    def from_text(cls, value: str) -> AtxyPageAddress:
        """Parse the canonical five-character ``A.TXY`` representation."""

        if not isinstance(value, str):
            raise TypeError("value must be a string")
        normalized = value.upper()
        if len(normalized) != 5 or normalized[1] != ".":
            raise ValueError("ATXY text must use the five-character A.TXY form")
        try:
            fields = tuple(_DIGITS.index(character) for character in normalized[0] + normalized[2:])
        except ValueError as error:
            raise ValueError("ATXY text contains a non-base36 character") from error
        return cls(*fields)

    @property
    def layer_index(self) -> int:
        return self.a

    @property
    def page_id(self) -> int:
        return (self.t * ATXY_RADIX + self.x) * ATXY_RADIX + self.y

    @property
    def packed(self) -> int:
        return (((self.a * ATXY_RADIX + self.t) * ATXY_RADIX + self.x) * ATXY_RADIX) + self.y

    @property
    def text(self) -> str:
        return f"{_DIGITS[self.a]}.{_DIGITS[self.t]}{_DIGITS[self.x]}{_DIGITS[self.y]}"


def token_capacity_per_layer(page_size: int) -> int:
    """Return the logical token capacity of one ATXY layer namespace."""

    page_size = _non_negative_int(page_size, "page_size")
    if page_size == 0:
        raise ValueError("page_size must be positive")
    return ATXY_PAGES_PER_LAYER * page_size
