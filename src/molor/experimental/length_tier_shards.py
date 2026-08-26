"""Deterministically partition pinned messages-only JSONL shards by rendered length.

This materializer is intentionally representation-agnostic: it never truncates, rewrites, or
inspects task-specific metadata because input rows may contain only ``messages``.  Every input
shard, tokenizer artifact set, and chat template is cryptographically pinned before any output
is published.  Direct, reasoning-trace, and tool conversations are rendered and emitted into
separate mode directories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Protocol, Sequence


SCHEMA_VERSION = "molor.length-tier-shards.v1"
PLAN_SCHEMA_VERSION = "molor.length-tier-plan.v1"
TEMPLATE_MODES = ("direct", "trace", "tool")
TIER_SPECS = (
    ("le-1024", 1, 1_024),
    ("1025-4096", 1_025, 4_096),
    ("4097-8192", 4_097, 8_192),
    ("8193-32768", 8_193, 32_768),
    ("deferred-gt32768", 32_769, None),
)

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
)
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})
_RESERVED_CHAT_MARKERS = ("<|im_start|>", "<|im_end|>")


class ChatTemplateTokenizer(Protocol):
    """Tokenizer surface used by the materializer and synthetic tests."""

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> object: ...


@dataclass(frozen=True, slots=True)
class InputShardDescriptor:
    """Immutable identity and render mode for one input JSONL shard."""

    path: Path
    sha256: str
    record_count: int
    template_mode: str


@dataclass(frozen=True, slots=True)
class TokenizerDescriptor:
    """Pinned local tokenizer artifact set and effective chat template."""

    path: Path
    aggregate_sha256: str
    chat_template_sha256: str


@dataclass(frozen=True, slots=True)
class LengthTierPlan:
    """Validated materialization plan loaded from JSON or constructed in Python."""

    tokenizer: TokenizerDescriptor
    inputs: tuple[InputShardDescriptor, ...]


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for ``value``."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Stream and hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return digest


def _chat_template_identity(root: Path) -> dict[str, Any]:
    template_file = root / "chat_template.jinja"
    if template_file.is_file():
        raw = template_file.read_bytes()
        if not raw.strip():
            raise ValueError(f"chat template is empty: {template_file}")
        return {
            "source": "chat_template.jinja",
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        }

    config_file = root / "tokenizer_config.json"
    if not config_file.is_file():
        raise ValueError(
            "local tokenizer must provide chat_template.jinja or tokenizer_config.json"
        )
    try:
        config = json.loads(config_file.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tokenizer_config.json: {config_file}") from exc
    template = config.get("chat_template") if isinstance(config, dict) else None
    if not isinstance(template, str) or not template.strip():
        raise ValueError("tokenizer_config.json must contain a non-empty chat_template string")
    raw = template.encode("utf-8")
    return {
        "source": "tokenizer_config.json:chat_template",
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
    }


def tokenizer_identity(path: str | Path) -> dict[str, Any]:
    """Return a stable identity for a local tokenizer and its effective chat template."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local tokenizer directory does not exist: {root}")
    artifacts: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for name in _TOKENIZER_ARTIFACTS:
        artifact = root / name
        if not artifact.is_file():
            continue
        digest = sha256_file(artifact)
        artifacts.append({"path": name, "sha256": digest, "bytes": artifact.stat().st_size})
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not artifacts:
        raise ValueError(f"no tokenizer artifacts found in {root}")
    return {
        "path": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "chat_template": _chat_template_identity(root),
        "artifacts": artifacts,
    }


def _parse_input_descriptor(value: object, *, base_dir: Path) -> InputShardDescriptor:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "record_count",
        "template_mode",
    }:
        raise ValueError(
            "each input descriptor must contain exactly path, sha256, record_count, "
            "and template_mode"
        )
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("input descriptor path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    record_count = value["record_count"]
    if type(record_count) is not int or record_count < 0:
        raise ValueError("input descriptor record_count must be a non-negative integer")
    mode = value["template_mode"]
    if mode not in TEMPLATE_MODES:
        raise ValueError(f"template_mode must be one of {TEMPLATE_MODES!r}")
    return InputShardDescriptor(
        path=path.resolve(),
        sha256=_checked_sha256(value["sha256"], field="input sha256"),
        record_count=record_count,
        template_mode=mode,
    )


def _parse_tokenizer_descriptor(value: object, *, base_dir: Path) -> TokenizerDescriptor:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "aggregate_sha256",
        "chat_template_sha256",
    }:
        raise ValueError(
            "tokenizer descriptor must contain exactly path, aggregate_sha256, "
            "and chat_template_sha256"
        )
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("tokenizer descriptor path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return TokenizerDescriptor(
        path=path.resolve(),
        aggregate_sha256=_checked_sha256(
            value["aggregate_sha256"], field="tokenizer aggregate_sha256"
        ),
        chat_template_sha256=_checked_sha256(
            value["chat_template_sha256"], field="tokenizer chat_template_sha256"
        ),
    )


def load_plan(path: str | Path) -> LengthTierPlan:
    """Load and strictly validate a length-tier plan."""

    plan_path = Path(path).resolve()
    try:
        document = json.loads(plan_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid length-tier plan JSON: {plan_path}") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "tokenizer",
        "inputs",
    }:
        raise ValueError("plan must contain exactly schema_version, tokenizer, and inputs")
    if document["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {PLAN_SCHEMA_VERSION!r}")
    raw_inputs = document["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("plan inputs must be a non-empty list")
    inputs = tuple(_parse_input_descriptor(item, base_dir=plan_path.parent) for item in raw_inputs)
    if len({descriptor.path for descriptor in inputs}) != len(inputs):
        raise ValueError("input descriptor paths must be unique")
    return LengthTierPlan(
        tokenizer=_parse_tokenizer_descriptor(document["tokenizer"], base_dir=plan_path.parent),
        inputs=inputs,
    )


def _verify_plan(plan: LengthTierPlan) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokenizer_receipt = tokenizer_identity(plan.tokenizer.path)
    if tokenizer_receipt["aggregate_sha256"] != plan.tokenizer.aggregate_sha256:
        raise ValueError(
            "tokenizer aggregate SHA-256 mismatch: "
            f"expected {plan.tokenizer.aggregate_sha256}, "
            f"got {tokenizer_receipt['aggregate_sha256']}"
        )
    actual_template_sha = tokenizer_receipt["chat_template"]["sha256"]
    if actual_template_sha != plan.tokenizer.chat_template_sha256:
        raise ValueError(
            "chat template SHA-256 mismatch: "
            f"expected {plan.tokenizer.chat_template_sha256}, got {actual_template_sha}"
        )

    if not plan.inputs:
        raise ValueError("at least one input descriptor is required")
    if len({descriptor.path.resolve() for descriptor in plan.inputs}) != len(plan.inputs):
        raise ValueError("input descriptor paths must be unique")
    input_receipts: list[dict[str, Any]] = []
    for index, descriptor in enumerate(plan.inputs):
        path = descriptor.path.resolve()
        if descriptor.template_mode not in TEMPLATE_MODES:
            raise ValueError(f"template_mode must be one of {TEMPLATE_MODES!r}")
        expected = _checked_sha256(descriptor.sha256, field="input sha256")
        if type(descriptor.record_count) is not int or descriptor.record_count < 0:
            raise ValueError("input descriptor record_count must be a non-negative integer")
        if path.suffix.casefold() != ".jsonl" or not path.is_file():
            raise FileNotFoundError(f"pinned input JSONL does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"input SHA-256 mismatch for descriptor {index}: expected {expected}, got {actual}"
            )
        input_receipts.append(
            {
                "index": index,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expected_sha256": expected,
                "expected_record_count": descriptor.record_count,
                "template_mode": descriptor.template_mode,
                "hash_verified": True,
            }
        )
    return tokenizer_receipt, input_receipts


def _thinking_parts(content: str) -> tuple[str, str] | None:
    stripped = content.strip()
    if stripped.count("<think>") != 1 or stripped.count("</think>") != 1:
        return None
    if not stripped.startswith("<think>"):
        return None
    close_at = stripped.find("</think>")
    if close_at < len("<think>"):
        return None
    trace = stripped[len("<think>") : close_at]
    final = stripped[close_at + len("</think>") :]
    return trace, final


def _validate_role_order(messages: list[dict[str, str]], *, mode: str) -> str | None:
    index = 0
    while index < len(messages) and messages[index]["role"] == "system":
        index += 1
    turn = messages[index:]
    if not turn or turn[0]["role"] != "user" or turn[-1]["role"] != "assistant":
        return "invalid_conversation_boundaries"
    if any(message["role"] == "system" for message in turn):
        return "system_role_after_turn_start"

    if mode in {"direct", "trace"}:
        if any(message["role"] == "tool" for message in turn):
            return "tool_role_in_non_tool_mode"
        expected = "user"
        for message in turn:
            if message["role"] != expected:
                return "invalid_non_tool_role_order"
            expected = "assistant" if expected == "user" else "user"
    elif not any(message["role"] == "tool" for message in turn):
        return "tool_mode_without_tool_role"
    return None


def _validate_messages_record(
    value: object, *, mode: str
) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(value, dict) or set(value) != {"messages"}:
        return None, "not_exact_messages_only_object"
    raw_messages = value["messages"]
    if not isinstance(raw_messages, list) or not raw_messages:
        return None, "invalid_messages_container"

    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            return None, "invalid_message_object"
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in _VALID_ROLES:
            return None, "invalid_message_role"
        if not isinstance(content, str) or not content.strip():
            return None, "invalid_message_content"
        if any(marker in content for marker in _RESERVED_CHAT_MARKERS):
            return None, "reserved_chat_marker"
        messages.append({"role": role, "content": content})

    role_error = _validate_role_order(messages, mode=mode)
    if role_error is not None:
        return None, role_error

    target = messages[-1]["content"]
    thinking_parts = _thinking_parts(target)
    has_thinking_marker = "<think>" in target or "</think>" in target
    if mode == "trace":
        if thinking_parts is None:
            return None, "malformed_trace_target"
        trace, final = thinking_parts
        if not trace.strip() or not final.strip():
            return None, "empty_trace_or_final"
    elif mode == "direct" and has_thinking_marker:
        if thinking_parts is None:
            return None, "malformed_direct_thinking_wrapper"
        trace, final = thinking_parts
        if trace.strip() or not final.strip():
            return None, "trace_target_in_direct_mode"
    return messages, None


def _token_ids(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("chat template must return a non-empty flat token list")
    if any(type(token_id) is not int or token_id < 0 for token_id in value):
        raise ValueError("chat template returned invalid token IDs")
    return list(value)


def render_token_count(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
    *,
    template_mode: str,
) -> int:
    """Render one complete conversation without truncation and return its exact length."""

    if template_mode not in TEMPLATE_MODES:
        raise ValueError(f"template_mode must be one of {TEMPLATE_MODES!r}")
    token_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=template_mode == "trace",
            truncation=False,
            padding=False,
            return_tensors=None,
            return_dict=False,
        )
    )
    return len(token_ids)


def tier_for_token_count(token_count: int) -> str:
    """Map a positive rendered token count to its non-overlapping output tier."""

    if type(token_count) is not int or token_count <= 0:
        raise ValueError("token_count must be a positive integer")
    for name, lower, upper in TIER_SPECS:
        if token_count >= lower and (upper is None or token_count <= upper):
            return name
    raise AssertionError("length tier specification did not cover a positive token count")


def _write_json_line(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _output_receipts(
    root: Path,
    *,
    tier_counts: Mapping[tuple[str, str], int],
    tier_min: Mapping[tuple[str, str], int],
    tier_max: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for mode in TEMPLATE_MODES:
        for tier, _, _ in TIER_SPECS:
            relative = Path(mode) / f"{tier}.jsonl"
            path = root / relative
            key = (mode, tier)
            receipts.append(
                {
                    "path": relative.as_posix(),
                    "template_mode": mode,
                    "tier": tier,
                    "records": tier_counts.get(key, 0),
                    "min_rendered_tokens": tier_min.get(key),
                    "max_rendered_tokens": tier_max.get(key),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    quarantine = root / "quarantine.jsonl"
    with quarantine.open("rb") as stream:
        quarantine_records = sum(1 for _ in stream)
    receipts.append(
        {
            "path": "quarantine.jsonl",
            "records": quarantine_records,
            "bytes": quarantine.stat().st_size,
            "sha256": sha256_file(quarantine),
            "content_free": True,
        }
    )
    return receipts


def materialize_length_tiers(
    *,
    plan: LengthTierPlan,
    output_dir: str | Path,
    tokenizer: ChatTemplateTokenizer,
) -> Path:
    """Materialize all pinned shards into deterministic mode/length files.

    The destination must not already exist.  Work is staged in a sibling temporary directory and
    is renamed into place only after every hash, declared record count, and output receipt passes.
    """

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_receipt, input_receipts = _verify_plan(plan)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    tier_counts: Counter[tuple[str, str]] = Counter()
    tier_min: dict[tuple[str, str], int] = {}
    tier_max: dict[tuple[str, str], int] = {}
    rejection_counts: Counter[str] = Counter()
    mode_input: Counter[str] = Counter()
    mode_accepted: Counter[str] = Counter()
    mode_quarantined: Counter[str] = Counter()
    actual_record_counts: list[int] = []

    try:
        with ExitStack() as stack:
            writers: dict[tuple[str, str], Any] = {}
            for mode in TEMPLATE_MODES:
                mode_dir = staging / mode
                mode_dir.mkdir(parents=True, exist_ok=True)
                for tier, _, _ in TIER_SPECS:
                    writers[(mode, tier)] = stack.enter_context(
                        (mode_dir / f"{tier}.jsonl").open("w", encoding="utf-8", newline="\n")
                    )
            quarantine = stack.enter_context(
                (staging / "quarantine.jsonl").open("w", encoding="utf-8", newline="\n")
            )

            for input_index, descriptor in enumerate(plan.inputs):
                actual_count = 0
                streamed_digest = hashlib.sha256()
                with descriptor.path.open("rb") as source:
                    for line_number, raw_line in enumerate(source, start=1):
                        streamed_digest.update(raw_line)
                        actual_count += 1
                        mode_input[descriptor.template_mode] += 1
                        fingerprint = sha256_bytes(raw_line.rstrip(b"\r\n"))
                        reason: str | None = None
                        record: object = None
                        try:
                            record = json.loads(raw_line.decode("utf-8"))
                        except UnicodeDecodeError:
                            reason = "invalid_utf8"
                        except json.JSONDecodeError:
                            reason = "invalid_json"

                        messages: list[dict[str, str]] | None = None
                        if reason is None:
                            messages, reason = _validate_messages_record(
                                record, mode=descriptor.template_mode
                            )
                        token_count: int | None = None
                        if reason is None and messages is not None:
                            try:
                                token_count = render_token_count(
                                    tokenizer,
                                    messages,
                                    template_mode=descriptor.template_mode,
                                )
                            except (TypeError, ValueError, RuntimeError):
                                reason = "tokenizer_render_error"

                        if reason is not None or messages is None or token_count is None:
                            rejection = reason or "internal_validation_error"
                            rejection_counts[rejection] += 1
                            mode_quarantined[descriptor.template_mode] += 1
                            _write_json_line(
                                quarantine,
                                {
                                    "input_index": input_index,
                                    "line_number": line_number,
                                    "record_sha256": fingerprint,
                                    "reason": rejection,
                                    "template_mode": descriptor.template_mode,
                                },
                            )
                            continue

                        tier = tier_for_token_count(token_count)
                        key = (descriptor.template_mode, tier)
                        _write_json_line(writers[key], {"messages": messages})
                        tier_counts[key] += 1
                        mode_accepted[descriptor.template_mode] += 1
                        tier_min[key] = min(tier_min.get(key, token_count), token_count)
                        tier_max[key] = max(tier_max.get(key, token_count), token_count)
                actual_record_counts.append(actual_count)
                if actual_count != descriptor.record_count:
                    raise ValueError(
                        f"input record-count mismatch for descriptor {input_index}: "
                        f"expected {descriptor.record_count}, got {actual_count}"
                    )
                streamed_sha256 = streamed_digest.hexdigest()
                if streamed_sha256 != descriptor.sha256.lower():
                    raise ValueError(
                        f"input changed while reading descriptor {input_index}: "
                        f"expected {descriptor.sha256.lower()}, got {streamed_sha256}"
                    )

        for receipt, actual_count in zip(input_receipts, actual_record_counts, strict=True):
            receipt["actual_record_count"] = actual_count
            receipt["record_count_verified"] = True
            receipt["stream_hash_verified"] = True

        output_receipts = _output_receipts(
            staging,
            tier_counts=tier_counts,
            tier_min=tier_min,
            tier_max=tier_max,
        )
        total_input = sum(actual_record_counts)
        total_accepted = sum(mode_accepted.values())
        total_quarantined = sum(mode_quarantined.values())
        if total_input != total_accepted + total_quarantined:
            raise AssertionError("aggregate materialization accounting is inconsistent")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "policy": {
                "input_schema": "exact messages-only JSON objects",
                "truncation": False,
                "malformed_records": "content-free quarantine receipt only",
                "mode_isolation": True,
                "tier_boundaries_inclusive": [
                    {"name": name, "minimum": lower, "maximum": upper}
                    for name, lower, upper in TIER_SPECS
                ],
                "template_modes": {
                    "direct": {"enable_thinking": False},
                    "trace": {"enable_thinking": True},
                    "tool": {"enable_thinking": False},
                },
            },
            "provenance": {
                "tokenizer": {
                    **tokenizer_receipt,
                    "expected_aggregate_sha256": plan.tokenizer.aggregate_sha256,
                    "expected_chat_template_sha256": plan.tokenizer.chat_template_sha256,
                    "hashes_verified": True,
                },
                "inputs": input_receipts,
            },
            "accounting": {
                "input_records": total_input,
                "accepted_records": total_accepted,
                "quarantined_records": total_quarantined,
                "by_mode": {
                    mode: {
                        "input": mode_input[mode],
                        "accepted": mode_accepted[mode],
                        "quarantined": mode_quarantined[mode],
                        "tiers": {tier: tier_counts[(mode, tier)] for tier, _, _ in TIER_SPECS},
                    }
                    for mode in TEMPLATE_MODES
                },
                "quarantine_reasons": dict(sorted(rejection_counts.items())),
                "fully_reconciled": True,
            },
            "outputs": output_receipts,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output)
        return output / "manifest.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_local_tokenizer(path: Path) -> ChatTemplateTokenizer:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - exercised only by the optional CLI
        raise RuntimeError("transformers is required to run the length-tier CLI") from exc
    return AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for a pinned length-tier plan."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    tokenizer = _load_local_tokenizer(plan.tokenizer.path)
    manifest = materialize_length_tiers(
        plan=plan,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
    )
    print(json.dumps({"manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
