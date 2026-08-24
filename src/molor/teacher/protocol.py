"""Machine-readable contract shared by the teacher HTTP API and CLI."""

from __future__ import annotations

from typing import Final


RUBRIC: Final[dict[str, object]] = {
    "version": "molor.teacher-rubric.v1",
    "scale": {
        "-2": "Critically wrong, unusable, or actively misleading.",
        "-1": "Material flaws require substantial correction.",
        "0": "Mixed, incomplete, or neither clearly good nor bad.",
        "1": "Good answer with only minor issues.",
        "2": "Excellent, accurate, complete, and well calibrated.",
    },
    "dimensions": {
        "accuracy": "Factual and technical correctness.",
        "instruction_following": "Satisfies the explicit request and constraints.",
        "reasoning": "Sound decomposition, assumptions, and conclusions.",
        "relevance": "Useful signal without unrelated material.",
        "style": "Clear, concise, appropriately structured communication.",
        "calibration": "Uncertainty and limitations are represented honestly.",
    },
    "teacher_guidance": [
        "Judge the response, not the identity or reputation of the student model.",
        "Use a corrected_answer when a concrete better target can be supplied.",
        "State specific errors in critique; unsupported praise is weak training signal.",
        "Only create a preference between generations produced from identical messages.",
        "Set confidence below 1 when verification is incomplete or the task is subjective.",
    ],
}


_TEACHER = {
    "type": "object",
    "required": ["id"],
    "properties": {
        "id": {"type": "string", "description": "Stable evaluator identity."},
        "model": {"type": "string", "description": "Exact teacher model/version if known."},
        "provider": {"type": "string"},
    },
}


_ROUTING = {
    "type": "object",
    "description": (
        "Request-time controls for the learned prompt router. If neither routing nor the legacy "
        "route object is supplied, automatic routing is used."
    ),
    "properties": {
        "mode": {"type": "string", "enum": ["auto", "manual"]},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 64},
        "temperature": {"type": "number", "exclusiveMinimum": 0},
        "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "null_bias": {"type": "number"},
        "max_prompt_tokens": {"type": "integer", "minimum": 1, "maximum": 65_536},
        "enabled_experts": {
            "type": "array",
            "uniqueItems": True,
            "items": {"type": "string"},
        },
        "manual_route": {
            "type": "object",
            "description": "Named base/expert weights used when mode is manual.",
            "additionalProperties": {"type": "number", "minimum": 0},
        },
    },
    "additionalProperties": False,
}


_ROUTING_DECISION = {
    "type": "object",
    "required": [
        "mode",
        "applied_route",
        "probabilities",
        "selected_routes",
        "confidence",
        "reason",
        "checkpoint_loaded",
    ],
    "properties": {
        "mode": {"type": "string", "enum": ["auto", "manual"]},
        "applied_route": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0},
        },
        "probabilities": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "selected_routes": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "checkpoint_loaded": {"type": "boolean"},
    },
}


TOOL_SCHEMAS: Final[list[dict[str, object]]] = [
    {
        "name": "ask_student",
        "description": (
            "Prompt the local frozen-base/LoRA student and persist one or more comparable "
            "candidate generations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["role", "content"],
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": ["system", "user", "assistant", "tool"],
                            },
                            "content": {"type": "string"},
                        },
                    },
                },
                "candidates": {"type": "integer", "minimum": 1, "maximum": 8},
                "persist": {
                    "type": "boolean",
                    "description": (
                        "Store generations in the feedback database. Benchmarks set this false "
                        "so bulk evaluation does not displace annotation work."
                    ),
                },
                "conversation_id": {"type": "string"},
                "teacher_session_id": {"type": "string"},
                "seed": {"type": "integer"},
                "route": {
                    "type": "object",
                    "description": (
                        "Legacy manual A/B route. Use routing.manual_route for stable named routes."
                    ),
                    "properties": {
                        "base": {"type": "number", "minimum": 0},
                        "expert_a": {"type": "number", "minimum": 0},
                        "expert_b": {"type": "number", "minimum": 0},
                    },
                },
                "routing": _ROUTING,
                "settings": {
                    "type": "object",
                    "properties": {
                        "max_new_tokens": {"type": "integer", "minimum": 1},
                        "temperature": {"type": "number", "minimum": 0},
                        "top_p": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                        "top_k": {"type": "integer", "minimum": 0},
                        "enable_thinking": {"type": "boolean"},
                        "reasoning_preset": {"type": "string"},
                        "system_instruction": {"type": "string"},
                        "context_token_budget": {"type": ["integer", "null"]},
                        "cache_mode": {
                            "type": "string",
                            "enum": ["dynamic", "offloaded", "disabled"],
                        },
                    },
                },
                "metadata": {"type": "object"},
            },
            "oneOf": [{"required": ["prompt"]}, {"required": ["messages"]}],
            "not": {"required": ["route", "routing"]},
        },
        "output_schema": {
            "type": "object",
            "required": [
                "conversation_id",
                "routing_policy",
                "candidates",
                "comparison_ready",
            ],
            "properties": {
                "conversation_id": {"type": "string"},
                "teacher_session_id": {"type": ["string", "null"]},
                "routing_policy": _ROUTING,
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "candidate_index",
                            "generation_id",
                            "response",
                            "telemetry",
                            "routing_decision",
                        ],
                        "properties": {
                            "candidate_index": {"type": "integer"},
                            "generation_id": {"type": ["string", "null"]},
                            "response": {"type": "string"},
                            "telemetry": {"type": "object"},
                            "routing_decision": _ROUTING_DECISION,
                        },
                    },
                },
                "comparison_ready": {"type": "boolean"},
            },
        },
    },
    {
        "name": "grade_student",
        "description": "Submit an independent rubric grade, critique, and optional correction.",
        "input_schema": {
            "type": "object",
            "required": ["generation_id", "teacher", "scores"],
            "properties": {
                "generation_id": {"type": "string"},
                "teacher": _TEACHER,
                "scores": {
                    "type": "object",
                    "required": ["overall"],
                    "additionalProperties": {
                        "type": "number",
                        "minimum": -2,
                        "maximum": 2,
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "critique": {"type": "string"},
                "corrected_answer": {"type": ["string", "null"]},
                "rubric_version": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    },
    {
        "name": "prefer_student_output",
        "description": "Choose the better of two outputs generated from identical messages.",
        "input_schema": {
            "type": "object",
            "required": [
                "chosen_generation_id",
                "rejected_generation_id",
                "teacher",
            ],
            "properties": {
                "chosen_generation_id": {"type": "string"},
                "rejected_generation_id": {"type": "string"},
                "teacher": _TEACHER,
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    },
    {
        "name": "next_student_review",
        "description": "Fetch the oldest generation not yet reviewed by this teacher.",
        "input_schema": {
            "type": "object",
            "required": ["teacher_id"],
            "properties": {"teacher_id": {"type": "string"}},
        },
    },
    {
        "name": "get_student_generation",
        "description": "Fetch a generation, all independent teacher annotations, and consensus.",
        "input_schema": {
            "type": "object",
            "required": ["generation_id"],
            "properties": {"generation_id": {"type": "string"}},
        },
    },
]


def capabilities() -> dict[str, object]:
    return {
        "name": "MoLoRA Teacher/Student API",
        "protocol_version": "molor.teacher-api.v1",
        "rubric": RUBRIC,
        "tools": TOOL_SCHEMAS,
        "transport": {
            "http": "JSON over loopback HTTP",
            "cli": "molor-teacher invoke <tool-name> --json '<payload>'",
        },
    }
