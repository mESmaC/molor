"""Validated configuration objects for the core MoLoR layers."""

from dataclasses import dataclass
import math
from typing import Any, Literal

from molor.identity import fingerprint_json

RoutingMode = Literal["sequence", "token"]


@dataclass(frozen=True, slots=True)
class LoRABankConfig:
    """Structure shared by every adapter expert in a bank."""

    num_experts: int
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    include_null_expert: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.num_experts, int)
            or isinstance(self.num_experts, bool)
            or self.num_experts < 1
        ):
            raise ValueError("num_experts must be at least 1")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 1:
            raise ValueError("rank must be at least 1")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError("alpha must be finite and positive")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if not isinstance(self.include_null_expert, bool):
            raise TypeError("include_null_expert must be boolean")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @property
    def num_routes(self) -> int:
        return self.num_experts + int(self.include_null_expert)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_experts": self.num_experts,
            "rank": self.rank,
            "alpha": float(self.alpha),
            "dropout": float(self.dropout),
            "include_null_expert": self.include_null_expert,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Configuration for a prompt- or token-level sparse router."""

    hidden_size: int
    num_experts: int
    top_k: int = 2
    mode: RoutingMode = "sequence"
    router_hidden_size: int | None = 256
    temperature: float = 1.0
    include_null_expert: bool = True
    normalize_top_k: bool = True
    uniform_top_k: bool = False
    straight_through: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_size, int)
            or isinstance(self.hidden_size, bool)
            or self.hidden_size < 1
        ):
            raise ValueError("hidden_size must be at least 1")
        if (
            not isinstance(self.num_experts, int)
            or isinstance(self.num_experts, bool)
            or self.num_experts < 1
        ):
            raise ValueError("num_experts must be at least 1")
        if self.mode not in ("sequence", "token"):
            raise ValueError("mode must be 'sequence' or 'token'")
        if self.router_hidden_size is not None and (
            not isinstance(self.router_hidden_size, int)
            or isinstance(self.router_hidden_size, bool)
            or self.router_hidden_size < 1
        ):
            raise ValueError("router_hidden_size must be a positive integer or None")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive")
        if (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or not 1 <= self.top_k <= self.num_routes
        ):
            raise ValueError("top_k must be between 1 and the number of routes")
        for field_name in (
            "include_null_expert",
            "normalize_top_k",
            "uniform_top_k",
            "straight_through",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be boolean")
        if self.straight_through and self.top_k != 1:
            raise ValueError("straight_through currently requires top_k=1")
        if self.straight_through and self.uniform_top_k:
            raise ValueError("straight_through and uniform_top_k are mutually exclusive")

    @property
    def num_routes(self) -> int:
        return self.num_experts + int(self.include_null_expert)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "mode": self.mode,
            "router_hidden_size": self.router_hidden_size,
            "temperature": float(self.temperature),
            "include_null_expert": self.include_null_expert,
            "normalize_top_k": self.normalize_top_k,
            "uniform_top_k": self.uniform_top_k,
            "straight_through": self.straight_through,
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())
