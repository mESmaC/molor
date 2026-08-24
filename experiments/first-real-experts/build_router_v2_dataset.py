"""Build a balanced five-route router corpus with the verified code expert.

The four existing v1 route splits are preserved as source partitions and downsampled to
the number of available verified-code prompts in each split.  Verified-code examples are
regenerated from ``build_verified_embedded_code.py`` and split by semantic group, so the
fenced/raw variants of one implementation task can never cross partitions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import unicodedata
from typing import Any, Iterable

from build_verified_embedded_code import build_examples


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = EXPERIMENT_ROOT / "data"
CURRENT_ROUTE_IDS = (
    "embedded-knowledge-v1",
    "ocb-can-verified-v1",
    "software-architecture-v1",
    "base/null",
)
CODE_ROUTE_ID = "verified-embedded-code-v1"
ROUTE_IDS = (*CURRENT_ROUTE_IDS[:-1], CODE_ROUTE_ID, CURRENT_ROUTE_IDS[-1])
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class Candidate:
    prompt: str
    normalized_prompt: str
    route_id: str
    split: str
    semantic_group: str
    source: str
    family: str | None = None

    def output_row(self) -> dict[str, str]:
        return {"prompt": self.prompt, "route_id": self.route_id}


def _normalize_prompt(prompt: str) -> str:
    value = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(value.split()).strip()


def _stable_hash(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _split_path(prefix: Path, split: str) -> Path:
    return prefix.with_name(f"{prefix.name}.{split}.jsonl")


def _report_path(prefix: Path) -> Path:
    return prefix.with_name(f"{prefix.name}.report.json")


def _load_v1_contract(prefix: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    report_path = _report_path(prefix)
    if not report_path.is_file():
        raise FileNotFoundError(f"v1 router report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid v1 router report: {report_path}") from exc
    if report.get("route_ids") != list(CURRENT_ROUTE_IDS):
        raise ValueError("v1 router report has an unexpected route axis")
    for field_name in ("normalized_prompt_overlap", "group_overlap"):
        overlap = report.get(field_name)
        if not isinstance(overlap, dict) or any(overlap.values()):
            raise ValueError(f"v1 router report does not prove zero {field_name}")

    outputs = report.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("v1 router report has no output identities")
    paths: dict[str, Path] = {}
    for split in SPLITS:
        path = _split_path(prefix, split).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"v1 router split does not exist: {path}")
        expected = outputs.get(split)
        if not isinstance(expected, dict) or expected.get("sha256") != _sha256_file(path):
            raise ValueError(f"v1 router split differs from its report: {path}")
        paths[split] = path
    return report, paths


def _load_v1_rows(paths: dict[str, Path]) -> dict[str, list[Candidate]]:
    results = {split: [] for split in SPLITS}
    for split in SPLITS:
        path = paths[split]
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict) or set(raw) != {"prompt", "route_id"}:
                        raise ValueError("row must contain exactly prompt and route_id")
                    prompt = raw["prompt"]
                    route_id = raw["route_id"]
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError("prompt must be non-empty text")
                    if route_id not in CURRENT_ROUTE_IDS:
                        raise ValueError(f"unknown v1 route {route_id!r}")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                normalized = _normalize_prompt(prompt)
                results[split].append(
                    Candidate(
                        prompt=prompt.strip(),
                        normalized_prompt=normalized,
                        route_id=route_id,
                        split=split,
                        # The source report, rather than this local identity, proves that the
                        # original semantic groups do not overlap between v1 partitions.
                        semantic_group=f"v1-preserved/{split}/{_stable_hash(normalized)}",
                        source="router-bootstrap-v1",
                    )
                )
    return results


def _verified_code_rows(
    *,
    generator_seed: int,
    split_seed: int,
) -> dict[str, list[Candidate]]:
    generated = build_examples(generator_seed)
    by_group: dict[str, list[Any]] = defaultdict(list)
    for example in generated:
        by_group[example.group].append(example)
    if len(generated) != sum(len(rows) for rows in by_group.values()):
        raise AssertionError("verified-code grouping lost examples")
    for group, rows in by_group.items():
        if len({row.family for row in rows}) != 1:
            raise ValueError(f"verified-code group spans multiple families: {group}")

    ordered_groups = sorted(by_group, key=lambda group: _stable_hash(split_seed, group))
    validation_groups = round(len(ordered_groups) * 0.10)
    test_groups = round(len(ordered_groups) * 0.10)
    train_groups = len(ordered_groups) - validation_groups - test_groups
    boundaries = {
        "train": (0, train_groups),
        "validation": (train_groups, train_groups + validation_groups),
        "test": (train_groups + validation_groups, len(ordered_groups)),
    }
    results = {split: [] for split in SPLITS}
    for split, (start, stop) in boundaries.items():
        for group in ordered_groups[start:stop]:
            for example in by_group[group]:
                prompt = example.prompt.strip()
                results[split].append(
                    Candidate(
                        prompt=prompt,
                        normalized_prompt=_normalize_prompt(prompt),
                        route_id=CODE_ROUTE_ID,
                        split=split,
                        semantic_group=f"verified-code/{group}",
                        source="build_verified_embedded_code.py",
                        family=example.family,
                    )
                )
    return results


def _assert_unique_prompts(candidates: Iterable[Candidate]) -> None:
    seen: dict[str, Candidate] = {}
    for candidate in candidates:
        previous = seen.get(candidate.normalized_prompt)
        if previous is not None:
            raise ValueError(
                "duplicate normalized prompt across router inputs: "
                f"{previous.source}/{previous.split} and {candidate.source}/{candidate.split}"
            )
        seen[candidate.normalized_prompt] = candidate


def _select_balanced(
    v1_rows: dict[str, list[Candidate]],
    code_rows: dict[str, list[Candidate]],
    *,
    seed: int,
) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, int]]]:
    selected: dict[str, list[Candidate]] = {}
    balance: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        by_route: dict[str, list[Candidate]] = {route_id: [] for route_id in ROUTE_IDS}
        for candidate in (*v1_rows[split], *code_rows[split]):
            by_route[candidate.route_id].append(candidate)
        available = {route_id: len(rows) for route_id, rows in by_route.items()}
        target = min(available.values())
        if target < 1:
            raise ValueError(f"one or more routes have no {split} examples: {available}")
        rows: list[Candidate] = []
        for route_id in ROUTE_IDS:
            ordered = sorted(
                by_route[route_id],
                key=lambda row: _stable_hash(seed, split, route_id, row.normalized_prompt),
            )
            rows.extend(ordered[:target])
        rows.sort(
            key=lambda row: _stable_hash(
                seed,
                "output",
                split,
                row.route_id,
                row.normalized_prompt,
            )
        )
        selected[split] = rows
        balance[split] = dict(Counter(row.route_id for row in rows))
        if set(balance[split].values()) != {target}:
            raise AssertionError(f"router v2 {split} split is not route-balanced")
    return selected, balance


def _write_jsonl(path: Path, rows: Iterable[Candidate]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in rows:
            line = json.dumps(candidate.output_row(), ensure_ascii=False, separators=(",", ":"))
            stream.write(line + "\n")
            digest.update((line + "\n").encode("utf-8"))
            count += 1
    return count, digest.hexdigest()


def build_router_v2_dataset(
    *,
    v1_prefix: Path,
    output_prefix: Path,
    generator_seed: int = 9071,
    split_seed: int = 211,
    selection_seed: int = 223,
) -> dict[str, Any]:
    v1_report, v1_paths = _load_v1_contract(v1_prefix.resolve())
    v1_rows = _load_v1_rows(v1_paths)
    code_rows = _verified_code_rows(
        generator_seed=generator_seed,
        split_seed=split_seed,
    )
    _assert_unique_prompts(
        candidate
        for split in SPLITS
        for candidate in (*v1_rows[split], *code_rows[split])
    )
    selected, balance = _select_balanced(v1_rows, code_rows, seed=selection_seed)

    prompt_sets = {
        split: {candidate.normalized_prompt for candidate in selected[split]}
        for split in SPLITS
    }
    group_sets = {
        split: {candidate.semantic_group for candidate in selected[split]}
        for split in SPLITS
    }
    prompt_overlap = {
        f"{left}/{right}": len(prompt_sets[left].intersection(prompt_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    group_overlap = {
        f"{left}/{right}": len(group_sets[left].intersection(group_sets[right]))
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    if any(prompt_overlap.values()) or any(group_overlap.values()):
        raise AssertionError("router v2 construction leaked prompts or semantic groups")

    outputs: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = _split_path(output_prefix, split).resolve()
        rows, digest = _write_jsonl(path, selected[split])
        outputs[split] = {"path": str(path), "rows": rows, "sha256": digest}

    code_family_counts = {
        split: dict(
            sorted(
                Counter(
                    row.family for row in code_rows[split] if row.family is not None
                ).items()
            )
        )
        for split in SPLITS
    }
    report = {
        "schema_version": "molor.router-bootstrap-dataset.v2",
        "route_ids": list(ROUTE_IDS),
        "seeds": {
            "verified_code_generator": generator_seed,
            "verified_code_split": split_seed,
            "balanced_selection": selection_seed,
        },
        "split_contract": {
            "v1": "original reported semantic-group-disjoint partitions preserved",
            "verified_code": (
                "generated semantic groups assigned 80/10/10 by stable hash order; "
                "all prompt variants from one group stay together"
            ),
        },
        "v1_source": {
            "report": str(_report_path(v1_prefix).resolve()),
            "report_sha256": _sha256_file(_report_path(v1_prefix).resolve()),
            "schema_version": v1_report.get("schema_version"),
            "splits": {
                split: {
                    "path": str(v1_paths[split]),
                    "sha256": _sha256_file(v1_paths[split]),
                    "rows": len(v1_rows[split]),
                }
                for split in SPLITS
            },
        },
        "verified_code_source": {
            "generator": str((EXPERIMENT_ROOT / "build_verified_embedded_code.py").resolve()),
            "generator_sha256": _sha256_file(
                EXPERIMENT_ROOT / "build_verified_embedded_code.py"
            ),
            "rows": {split: len(code_rows[split]) for split in SPLITS},
            "semantic_groups": {
                split: len({row.semantic_group for row in code_rows[split]})
                for split in SPLITS
            },
            "families": code_family_counts,
        },
        "balance": balance,
        "normalized_prompt_overlap": prompt_overlap,
        "semantic_group_overlap": group_overlap,
        "duplicate_normalized_prompts": 0,
        "outputs": outputs,
    }
    report_path = _report_path(output_prefix).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v1",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DATA_ROOT / "router-bootstrap-v2",
    )
    parser.add_argument("--generator-seed", type=int, default=9071)
    parser.add_argument("--split-seed", type=int, default=211)
    parser.add_argument("--selection-seed", type=int, default=223)
    args = parser.parse_args()
    report = build_router_v2_dataset(
        v1_prefix=args.v1_prefix,
        output_prefix=args.output_prefix,
        generator_seed=args.generator_seed,
        split_seed=args.split_seed,
        selection_seed=args.selection_seed,
    )
    print(
        json.dumps(
            {
                "route_ids": report["route_ids"],
                "balance": report["balance"],
                "outputs": report["outputs"],
                "report": report["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
