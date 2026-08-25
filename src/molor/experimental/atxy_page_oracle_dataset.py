"""Deterministic synthetic datasets for the ATXY exact-payload page oracle.

This module performs tokenizer-only rendering and validation.  It never loads
model weights or runs model inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_SCHEMA = "molor.atxy-page-oracle-development-dataset.v1"
DATASET_STATUS = "development-only"
FULL_TOKEN_COUNT = 32_768
CHECKPOINTS = (4_096, 8_192, 16_384, 32_767, 32_768)
DEFAULT_MODEL = Path("models/Qwen3-1.7B")
FULL_MANIFEST_NAME = "development-manifest.json"
SMOKE_MANIFEST_NAME = "smoke-manifest.json"
CHECKSUM_NAME = "SHA256SUMS"


@dataclass(frozen=True, slots=True)
class CaseRecipe:
    case_id: str
    case_number: int
    domain_order: tuple[str, str, str, str]
    anchors: tuple[str, str, str, str, str]


@dataclass(frozen=True, slots=True)
class MarkerRecipe:
    checkpoint: int
    anchor: str
    support: str
    distractors: tuple[str, str]
    query_text: str


@dataclass(frozen=True, slots=True)
class RenderedCheckpoint:
    token_count: int
    support_token_span: tuple[int, int]
    support_token_ids: tuple[int, ...]
    distractor_token_ids: tuple[tuple[int, ...], tuple[int, ...]]
    query_end_character: int


@dataclass(frozen=True, slots=True)
class RenderedCase:
    case_id: str
    source_text: str
    token_ids: tuple[int, ...]
    checkpoints: tuple[RenderedCheckpoint, ...]


@dataclass(frozen=True, slots=True)
class DatasetAssets:
    files: Mapping[str, bytes]
    full_cases: tuple[RenderedCase, ...]
    smoke_case: RenderedCase


CASE_RECIPES = (
    CaseRecipe(
        case_id="mixed-development-01",
        case_number=1,
        domain_order=("software", "embedded", "finance", "terminal"),
        anchors=("quartz", "ember", "amber", "azure", "teal"),
    ),
    CaseRecipe(
        case_id="mixed-development-02",
        case_number=2,
        domain_order=("embedded", "terminal", "software", "finance"),
        anchors=("river", "maple", "silver", "cedar", "ivory"),
    ),
    CaseRecipe(
        case_id="mixed-development-03",
        case_number=3,
        domain_order=("finance", "software", "terminal", "embedded"),
        anchors=("violet", "copper", "coral", "golden", "pearl"),
    ),
    CaseRecipe(
        case_id="mixed-development-04",
        case_number=4,
        domain_order=("terminal", "finance", "embedded", "software"),
        anchors=("winter", "marble", "sunset", "olive", "bronze"),
    ),
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encoded(tokenizer: Any, text: str) -> tuple[int, ...]:
    output = tokenizer(text, add_special_tokens=True)
    if not isinstance(output, Mapping) or "input_ids" not in output:
        raise ValueError("tokenizer output must contain input_ids")
    token_ids = output["input_ids"]
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids
    ):
        raise ValueError("tokenizer input_ids must be a flat integer list")
    return tuple(token_ids)


def _encoded_with_offsets(
    tokenizer: Any, text: str
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    output = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    if not isinstance(output, Mapping):
        raise ValueError("tokenizer output must be a mapping")
    token_ids = output.get("input_ids")
    offsets = output.get("offset_mapping")
    if (
        not isinstance(token_ids, list)
        or not isinstance(offsets, list)
        or len(token_ids) != len(offsets)
        or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in token_ids
        )
        or any(
            not isinstance(offset, Sequence)
            or isinstance(offset, (str, bytes))
            or len(offset) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offset)
            for offset in offsets
        )
    ):
        raise ValueError("fast tokenizer returned unsupported IDs or offsets")
    return tuple(token_ids), tuple((int(offset[0]), int(offset[1])) for offset in offsets)


def _domain_line(domain: str, *, case_number: int, segment: int, row: int) -> str:
    serial = case_number * 100_000 + segment * 10_000 + row
    if domain == "software":
        return (
            f"SW synthetic={serial:06d} module=atlas_{row % 29:02d} patch={serial % 997:03d} "
            f"test_shard={row % 17:02d} result={'PASS' if row % 7 else 'RETRY'} "
            f"latency_ms={11 + (row * 13) % 89} owner=team_{(row + case_number) % 9}.\n"
        )
    if domain == "embedded":
        return (
            f"CAN synthetic={serial:06d} bus={row % 4} arbitration=0x{(0x120 + row * 7) % 0x7FF:03X} "
            f"payload={serial % 65536:04X}{(serial * 3) % 65536:04X} counter={row % 16:02d} "
            f"crc={'OK' if row % 11 else 'REPLAY'} temperature_c={20 + row % 47}.\n"
        )
    if domain == "finance":
        return (
            f"FIN synthetic={serial:06d} account=SIM-{case_number:02d}-{row % 41:02d} "
            f"lot={1 + row % 23} notional_units={1000 + (row * 137) % 9000} "
            f"fee_basis_points={1 + row % 8} risk_band=R{row % 6} settlement=D+{1 + row % 3}.\n"
        )
    if domain == "terminal":
        return (
            f"TERM synthetic={serial:06d} session=LAB-{case_number:02d}-{row:05d} "
            f"request=inspect_node_{row % 31:02d} response={'READY' if row % 5 else 'QUEUED'} "
            f"ticket=DEV-{serial % 10000:04d} channel=simulated_{row % 7} retries={row % 3}.\n"
        )
    raise ValueError(f"unknown synthetic domain: {domain}")


def _marker_recipe(case: CaseRecipe, checkpoint: int, anchor: str) -> MarkerRecipe:
    value_number = (case.case_number * 37 + checkpoint // 64) % 997
    correct_value = f"VX-{case.case_number:02d}-{checkpoint:05d}-{value_number:03d}"
    wrong_value = f"VX-{case.case_number:02d}-{checkpoint:05d}-{(value_number + 2) % 997:03d}"
    prefix = f"[ATXY record case={case.case_number:02d} checkpoint={checkpoint:05d}"
    support = f"{prefix} state=ACTIVE anchor={anchor} value={correct_value}]"
    distractors = (
        f"{prefix} state=ARCHIVED anchor={anchor} value={wrong_value}]",
        f"{prefix} state=ACTIVE anchor={anchor}x value={correct_value}]",
    )
    query_text = (
        f"ATXY checkpoint query case {case.case_number:02d} target {checkpoint:05d}: "
        f"return the ACTIVE exact-anchor value for {anchor}"
    )
    return MarkerRecipe(
        checkpoint=checkpoint,
        anchor=anchor,
        support=support,
        distractors=distractors,
        query_text=query_text,
    )


def _single_trailer(marker: MarkerRecipe) -> str:
    return (
        f"{marker.distractors[0]}\n{marker.support}\n{marker.distractors[1]}\n{marker.query_text}"
    )


def _dual_trailer(
    first: MarkerRecipe, second: MarkerRecipe, case_number: int
) -> tuple[str, str, str]:
    prefix = (
        f"{first.distractors[0]}\n{first.support}\n{first.distractors[1]}\n"
        f"{second.distractors[0]}\n{second.support}\n{second.distractors[1]}\n"
        f"ATXY dual checkpoint query case {case_number:02d}: "
        "resolve ACTIVE exact-anchor records in order:"
    )
    first_query = f"{prefix} {first.anchor}"
    second_query = f"{first_query} {second.anchor}"
    return second_query, first_query, second_query


def _fit_segment(
    tokenizer: Any,
    *,
    prefix: str,
    header: str,
    trailer: str,
    target: int,
    domain: str,
    case_number: int,
    segment: int,
) -> str:
    body = prefix + header
    next_row = 0
    reserve = 96
    batch_size = 16
    while True:
        batch = "".join(
            _domain_line(
                domain,
                case_number=case_number,
                segment=segment,
                row=next_row + offset,
            )
            for offset in range(batch_size)
        )
        if len(_encoded(tokenizer, body + batch + "\n" + trailer)) > target - reserve:
            break
        body += batch
        next_row += batch_size

    while True:
        line = _domain_line(
            domain,
            case_number=case_number,
            segment=segment,
            row=next_row,
        )
        if len(_encoded(tokenizer, body + line + "\n" + trailer)) > target - reserve:
            break
        body += line
        next_row += 1

    def candidate(padding_tokens: int) -> str:
        return body + (" x" * padding_tokens) + "\n" + trailer

    if len(_encoded(tokenizer, candidate(0))) > target:
        raise RuntimeError(f"synthetic segment {segment} overfilled checkpoint {target}")
    low = 0
    high = max(1, target - len(_encoded(tokenizer, candidate(0))) + 32)
    while len(_encoded(tokenizer, candidate(high))) < target:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(_encoded(tokenizer, candidate(middle))) < target:
            low = middle + 1
        else:
            high = middle
    rendered = candidate(low)
    actual = len(_encoded(tokenizer, rendered))
    if actual != target:
        raise RuntimeError(
            f"could not fit synthetic segment {segment} exactly: wanted {target}, got {actual}"
        )
    return rendered


def _token_span(
    *,
    text: str,
    token_ids: tuple[int, ...],
    offsets: tuple[tuple[int, int], ...],
    marker: str,
) -> tuple[int, int]:
    if text.count(marker) != 1:
        raise RuntimeError("support or distractor marker is not textually unique")
    character_start = text.index(marker)
    character_stop = character_start + len(marker)
    indices = [
        index
        for index, (start, stop) in enumerate(offsets)
        if stop > character_start and start < character_stop
    ]
    if (
        not indices
        or indices != list(range(indices[0], indices[-1] + 1))
        or any(stop <= start for start, stop in (offsets[index] for index in indices))
    ):
        raise RuntimeError("marker does not align to an exact tokenizer span")
    covered_start = offsets[indices[0]][0]
    covered_stop = offsets[indices[-1]][1]
    if text[covered_start:character_start].strip() or text[character_stop:covered_stop].strip():
        raise RuntimeError("marker boundary shares a token with non-whitespace text")
    span = (indices[0], indices[-1] + 1)
    needle = token_ids[span[0] : span[1]]
    occurrences = sum(
        token_ids[index : index + len(needle)] == needle
        for index in range(len(token_ids) - len(needle) + 1)
    )
    if occurrences != 1:
        raise RuntimeError("support or distractor token subsequence is ambiguous")
    return span


def _validate_case(
    tokenizer: Any,
    *,
    case: CaseRecipe,
    text: str,
    markers: tuple[MarkerRecipe, ...],
    query_texts: tuple[str, ...],
) -> RenderedCase:
    token_ids, offsets = _encoded_with_offsets(tokenizer, text)
    if len(token_ids) != FULL_TOKEN_COUNT:
        raise RuntimeError(f"{case.case_id} has {len(token_ids)} tokens, not {FULL_TOKEN_COUNT}")
    checkpoints: list[RenderedCheckpoint] = []
    support_spans: list[tuple[int, int]] = []
    for checkpoint, marker, query_text in zip(CHECKPOINTS, markers, query_texts, strict=True):
        support_span = _token_span(
            text=text,
            token_ids=token_ids,
            offsets=offsets,
            marker=marker.support,
        )
        distractor_spans = tuple(
            _token_span(
                text=text,
                token_ids=token_ids,
                offsets=offsets,
                marker=distractor,
            )
            for distractor in marker.distractors
        )
        if support_span[1] > checkpoint - 1:
            raise RuntimeError(f"{case.case_id} checkpoint {checkpoint} support is not causal")
        support_ids = token_ids[support_span[0] : support_span[1]]
        distractor_ids = tuple(token_ids[start:stop] for start, stop in distractor_spans)
        for distractor in distractor_ids:
            similarity = SequenceMatcher(None, support_ids, distractor, autojunk=False).ratio()
            if support_ids == distractor or similarity < 0.72:
                raise RuntimeError(
                    f"{case.case_id} checkpoint {checkpoint} lacks a close distinct distractor"
                )

        if text.count(query_text) != 1:
            raise RuntimeError(f"{case.case_id} checkpoint {checkpoint} query is not unique")
        query_end = text.index(query_text) + len(query_text)
        ending_tokens = [
            index for index, (start, stop) in enumerate(offsets) if start < query_end <= stop
        ]
        if len(ending_tokens) != 1 or ending_tokens[0] != checkpoint - 1:
            raise RuntimeError(
                f"{case.case_id} query does not end at checkpoint token {checkpoint}"
            )
        anchor_ids = _encoded(tokenizer, f" {marker.anchor}")
        if len(anchor_ids) != 1 or token_ids[checkpoint - 1] != anchor_ids[0]:
            raise RuntimeError(f"{case.case_id} checkpoint anchor is not one stable token")

        support_spans.append(support_span)
        checkpoints.append(
            RenderedCheckpoint(
                token_count=checkpoint,
                support_token_span=support_span,
                support_token_ids=support_ids,
                distractor_token_ids=(distractor_ids[0], distractor_ids[1]),
                query_end_character=query_end,
            )
        )
    if len(set(support_spans)) != len(CHECKPOINTS):
        raise RuntimeError(f"{case.case_id} support spans are not unique across checkpoints")
    if _encoded(tokenizer, text) != token_ids:
        raise RuntimeError(f"{case.case_id} failed exact tokenizer round-trip validation")
    return RenderedCase(
        case_id=case.case_id,
        source_text=text,
        token_ids=token_ids,
        checkpoints=tuple(checkpoints),
    )


def render_case(tokenizer: Any, case: CaseRecipe) -> RenderedCase:
    """Render and validate one exact 32,768-token heterogeneous case."""

    markers = tuple(
        _marker_recipe(case, checkpoint, anchor)
        for checkpoint, anchor in zip(CHECKPOINTS, case.anchors, strict=True)
    )
    text = ""
    query_texts: list[str] = []
    for segment, checkpoint in enumerate(CHECKPOINTS[:3]):
        marker = markers[segment]
        text = _fit_segment(
            tokenizer,
            prefix=text,
            header=(
                f"\n\n=== SYNTHETIC ATXY CASE {case.case_number:02d} SEGMENT {segment + 1:02d} "
                f"DOMAIN {case.domain_order[segment].upper()} ===\n"
            ),
            trailer=_single_trailer(marker),
            target=checkpoint,
            domain=case.domain_order[segment],
            case_number=case.case_number,
            segment=segment + 1,
        )
        query_texts.append(marker.query_text)

    final_trailer, first_final_query, second_final_query = _dual_trailer(
        markers[3], markers[4], case.case_number
    )
    text = _fit_segment(
        tokenizer,
        prefix=text,
        header=(
            f"\n\n=== SYNTHETIC ATXY CASE {case.case_number:02d} SEGMENT 04 "
            f"DOMAIN {case.domain_order[3].upper()} ===\n"
        ),
        trailer=final_trailer,
        target=FULL_TOKEN_COUNT,
        domain=case.domain_order[3],
        case_number=case.case_number,
        segment=4,
    )
    query_texts.extend((first_final_query, second_final_query))
    return _validate_case(
        tokenizer,
        case=case,
        text=text,
        markers=markers,
        query_texts=tuple(query_texts),
    )


def _manifest_bytes(cases: Sequence[RenderedCase], paths: Mapping[str, str]) -> bytes:
    payload = {
        "schema_version": DATASET_SCHEMA,
        "status": DATASET_STATUS,
        "cases": [
            {
                "id": case.case_id,
                "path": paths[case.case_id],
                "sha256": _sha256(case.source_text.encode("utf-8")),
                "token_count": len(case.token_ids),
                "queries": [
                    {
                        "token_count": checkpoint.token_count,
                        "support_token_spans": [list(checkpoint.support_token_span)],
                    }
                    for checkpoint in case.checkpoints
                ],
            }
            for case in cases
        ],
    }
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def render_dataset(tokenizer: Any) -> DatasetAssets:
    """Render the full development corpus and its exact 4K smoke prefix."""

    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("dataset rendering requires a fast tokenizer with character offsets")
    full_cases = tuple(render_case(tokenizer, recipe) for recipe in CASE_RECIPES)
    first_case = full_cases[0]
    first_checkpoint = first_case.checkpoints[0]
    smoke_text = first_case.source_text[: first_checkpoint.query_end_character]
    smoke_ids = _encoded(tokenizer, smoke_text)
    if smoke_ids != first_case.token_ids[:4_096]:
        raise RuntimeError("smoke source is not the exact first causal checkpoint prefix")
    smoke_case = RenderedCase(
        case_id="mixed-development-01-smoke",
        source_text=smoke_text,
        token_ids=smoke_ids,
        checkpoints=(first_checkpoint,),
    )

    full_paths = {case.case_id: f"prompts/{case.case_id}.txt" for case in full_cases}
    smoke_paths = {smoke_case.case_id: "smoke-prompts/mixed-development-01-smoke.txt"}
    files: dict[str, bytes] = {
        path: case.source_text.encode("utf-8")
        for case in full_cases
        for path in (full_paths[case.case_id],)
    }
    files[smoke_paths[smoke_case.case_id]] = smoke_case.source_text.encode("utf-8")
    files[FULL_MANIFEST_NAME] = _manifest_bytes(full_cases, full_paths)
    files[SMOKE_MANIFEST_NAME] = _manifest_bytes((smoke_case,), smoke_paths)
    checksum_lines = [
        f"{_sha256(payload)}  {path}\n"
        for path, payload in sorted(files.items(), key=lambda item: item[0])
    ]
    files[CHECKSUM_NAME] = "".join(checksum_lines).encode("ascii")
    return DatasetAssets(files=files, full_cases=full_cases, smoke_case=smoke_case)


def write_assets_exclusive(output_directory: Path, assets: DatasetAssets) -> tuple[Path, ...]:
    """Create missing assets, accept byte-identical assets, and reject drift."""

    output = output_directory.expanduser().resolve()
    targets = {output / relative: payload for relative, payload in assets.files.items()}
    for path, payload in targets.items():
        if path.is_symlink():
            raise FileExistsError(f"refusing generated-asset symlink: {path}")
        if path.exists():
            if not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(
                    f"generated asset exists with different bytes; move it explicitly: {path}"
                )

    written: list[Path] = []
    for path, payload in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        try:
            with path.open("xb") as handle:
                handle.write(payload)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise
        written.append(path)

    for path, payload in targets.items():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"generated asset failed post-write verification: {path}")
    return tuple(written)


def _load_local_tokenizer(model_path: Path) -> Any:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"tokenizer model must be a local directory: {resolved}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        resolved,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )


def build_parser(default_output_directory: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=default_output_directory or Path("experiments/atxy-page-oracle-v1/dataset"),
    )
    return parser


def main(argv: Sequence[str] | None = None, *, default_output_directory: Path | None = None) -> int:
    args = build_parser(default_output_directory).parse_args(argv)
    tokenizer = _load_local_tokenizer(args.model)
    assets = render_dataset(tokenizer)
    written = write_assets_exclusive(args.output_directory, assets)
    for path, payload in sorted(assets.files.items()):
        state = (
            "created"
            if args.output_directory.expanduser().resolve() / path in written
            else "verified"
        )
        print(f"{state} {path} bytes={len(payload)} sha256={_sha256(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
