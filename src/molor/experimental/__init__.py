"""Small, experiment-first APIs for running mixtures of LoRA experts."""

from molor.experimental.autorouter import (
    AutoRouteDecision,
    AutoRouterPolicy,
    NULL_ROUTE_ID,
    PromptAutoRouter,
    ROUTER_CHECKPOINT_SCHEMA,
    base_fallback_decision,
    manual_route_decision,
)
from molor.experimental.hashed_kv import (
    AttentionMetrics,
    HashedKVConfig,
    HashedKVShadowResult,
    PageSelection,
    PageSpan,
    RandomHyperplaneSimHash,
    evaluate_hashed_kv,
    select_minmax_pages,
)
from molor.experimental.incremental_autorouter import IncrementalPromptAutoRouter
from molor.experimental.mixture import (
    BASE_ROUTE,
    QWEN3_TARGET_MODULES,
    CacheMode,
    ExpertConfig,
    GenerationSettings,
    GenerationStats,
    GenerationUpdate,
    MixtureEngine,
    MixtureLinear,
    RouteController,
)

__all__ = [
    "AutoRouteDecision",
    "AutoRouterPolicy",
    "AttentionMetrics",
    "BASE_ROUTE",
    "HashedKVConfig",
    "HashedKVShadowResult",
    "IncrementalPromptAutoRouter",
    "NULL_ROUTE_ID",
    "PromptAutoRouter",
    "PageSelection",
    "PageSpan",
    "QWEN3_TARGET_MODULES",
    "ROUTER_CHECKPOINT_SCHEMA",
    "CacheMode",
    "ExpertConfig",
    "GenerationSettings",
    "GenerationStats",
    "GenerationUpdate",
    "MixtureEngine",
    "MixtureLinear",
    "RouteController",
    "RandomHyperplaneSimHash",
    "base_fallback_decision",
    "evaluate_hashed_kv",
    "manual_route_decision",
    "select_minmax_pages",
]
