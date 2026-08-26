from __future__ import annotations

from html import escape
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from molor.experimental.business_reference_corpus import (
    BuildConfig,
    LEAVES,
    SELECTED_SOURCE_IDS,
    build_business_reference_corpus,
    load_business_receipt,
    main,
)


SOURCE_ROOTS = {
    "fdic-sba-money-smart-small-business": "fdic-money-smart-small-business",
    "gitlab-company-handbook": "gitlab-handbook",
    "onet-business-work-activities": "onet-31.0",
    "posthog-handbook-and-manual": "posthog-handbook",
    "sba-business-guide-and-content-api": "sba-business-guide",
    "yc-startup-library": "yc-startup-library",
}


def _pdf_bytes(text: str) -> bytes:
    from reportlab.pdfgen.canvas import Canvas

    stream = BytesIO()
    canvas = Canvas(stream)
    y = 800
    for line in text.splitlines():
        canvas.drawString(40, y, line[:100])
        y -= 18
    canvas.save()
    return stream.getvalue()


def _onet_zip_bytes() -> bytes:
    occupations = {
        "row": [
            {
                "onetsoc_code": "11-1011.00",
                "title": "Business Operations Manager",
                "description": (
                    "Plan and direct company operations, management workflows, staffing, "
                    "budgets, and organizational objectives."
                ),
            },
            {
                "onetsoc_code": "15-0000.00",
                "title": "Out of scope role",
                "description": "This occupation is intentionally outside selected groups.",
            },
        ]
    }
    tasks = {
        "row": [
            {
                "onetsoc_code": "11-1011.00",
                "task": "Coordinate business planning, hiring, procurement, and team workflows.",
            }
        ]
    }

    def element(name: str) -> dict[str, object]:
        return {"row": [{"onetsoc_code": "11-1011.00", "element_name": name}]}

    members = {
        "db_31_0_json/occupation_data.json": occupations,
        "db_31_0_json/task_statements.json": tasks,
        "db_31_0_json/work_activities.json": element("Coordinating the Work of Others"),
        "db_31_0_json/knowledge.json": element("Administration and Management"),
        "db_31_0_json/essential_skills.json": element("Resource Management"),
        "db_31_0_json/work_styles.json": element("Leadership Orientation"),
        "db_31_0_json/job_zones.json": {"row": []},
    }
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, json.dumps(value, sort_keys=True))
    return stream.getvalue()


def _presentation_bytes() -> bytes:
    slide = b"""<?xml version="1.0" encoding="UTF-8"?>
    <p:sld xmlns:p="urn:presentation" xmlns:a="urn:drawing">
      <p:cSld><a:t>Managing cash flow and business finance</a:t>
      <a:t>Use accounting, financial statements, budgets, bookkeeping, expense controls,
      and monthly forecasts to guide a small business through operating decisions.</a:t></p:cSld>
    </p:sld>"""
    core = b"""<?xml version="1.0" encoding="UTF-8"?>
    <cp:coreProperties xmlns:cp="urn:core" xmlns:dc="urn:title">
      <dc:title>Small business finance workshop</dc:title>
    </cp:coreProperties>"""
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("docProps/core.xml", core)
    return stream.getvalue()


def _yc_html() -> bytes:
    article = {
        "title": "How to build a seed round pitch deck",
        "content": (
            "Seed round fundraising requires a clear pitch deck. Explain the market, product, "
            "traction, team, funding target, investor fit, and how the company will use capital. "
            "A strong founder narrative should remain grounded in measurable startup progress."
        ),
        "transcript": None,
        "description": "A grounded guide to fundraising and investor communication.",
    }
    page = {"props": {"article": article}}
    return (
        '<html><head><meta name="csrf-token" content="fixture-token-never-emitted"></head>'
        f'<body><div data-page="{escape(json.dumps(page), quote=True)}"></div></body></html>'
    ).encode("utf-8")


def _build_fixture(tmp_path: Path, *, license_gate: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "business-sources-v1"
    artifact_rows: dict[str, list[dict[str, object]]] = {
        source_id: [] for source_id in SELECTED_SOURCE_IDS
    }

    def add(source_id: str, relative_path: str, payload: bytes) -> None:
        path = root / SOURCE_ROOTS[source_id] / Path(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifact_rows[source_id].append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )

    add(
        "fdic-sba-money-smart-small-business",
        "modules.csv",
        (
            "Topic,Grounding\nFinance,Accounting cash flow budgeting bookkeeping financial "
            "statements accounts payable and business finance controls for a small business.\n"
        ).encode(),
    )
    add(
        "fdic-sba-money-smart-small-business",
        "operations-guide.pdf",
        _pdf_bytes(
            "Small Business Operations Guide\n"
            "Plan repeatable management workflows, hiring, procurement, inventory management, "
            "and process improvement. Review operating plans with the team every month."
        ),
    )
    add(
        "fdic-sba-money-smart-small-business",
        "workshop-module.pdf",
        _presentation_bytes(),
    )
    duplicate_text = (
        "Public markets and investment analysis require evaluating securities, fixed income, "
        "portfolio management, market valuation, business fundamentals, and durable cash flows. "
        "Analysts compare risks and returns using documented assumptions and consistent methods."
    )
    add(
        "gitlab-company-handbook",
        "content/handbook/finance/investing-a.md",
        f"---\ntitle: Investment analysis\n---\n\n{duplicate_text}\n".encode(),
    )
    add(
        "gitlab-company-handbook",
        "content/handbook/finance/investing-b.md",
        f"---\ntitle: Alternate investment heading\n---\n\n  {duplicate_text}  \n".encode(),
    )
    add(
        "gitlab-company-handbook",
        "content/handbook/people/operations.md",
        (
            "# People operations\n\nBusiness operations includes hiring, recruiting, human "
            "resources, management, leadership, compliance, and repeatable workflows. Contact "
            "ops@example.com or 415-555-1212. api_key=abcdefghijk123456789"
        ).encode(),
    )
    add(
        "gitlab-company-handbook",
        "package.json",
        b'{"scripts":{"build":"fixture"}}\n',
    )
    add(
        "onet-business-work-activities",
        "db_31_0_json.zip",
        _onet_zip_bytes(),
    )
    add(
        "posthog-handbook-and-manual",
        "contents/founders/growth.md",
        (
            "---\ntitle: Product-led growth\n---\n\nA go-to-market strategy connects pricing, "
            "marketing, sales, customer acquisition, retention, conversion, and product-market "
            "fit. Teams measure the sales pipeline and churn before expanding channels."
        ).encode(),
    )
    near_text = (
        "Growth growth growth teams document customer acquisition, marketing, sales, pricing, "
        "retention, conversion, product-market fit, channel strategy, and measurable business "
        "outcomes before expanding a repeatable go-to-market program."
    )
    add(
        "posthog-handbook-and-manual",
        "contents/founders/near-growth-a.md",
        f"---\ntitle: Growth field guide A\n---\n\n{near_text}\n".encode(),
    )
    add(
        "posthog-handbook-and-manual",
        "contents/founders/near-growth-b.md",
        f"---\ntitle: Growth field guide B\n---\n\n{near_text} Growth.\n".encode(),
    )
    add(
        "sba-business-guide-and-content-api",
        "plan-your-business.html",
        (
            "<html><head><title>Plan your business</title><script>token='never'</script></head>"
            "<body><main><h1>Business planning</h1><p>An entrepreneur creates a business plan, "
            "tests a business model, studies competitive advantage, chooses incorporation, and "
            "builds a practical company strategy for a new small business.</p></main></body></html>"
        ).encode(),
    )
    add("yc-startup-library", "seed-round.html", _yc_html())

    sources: list[dict[str, object]] = []
    for source_id in SELECTED_SOURCE_IDS:
        artifacts = sorted(artifact_rows[source_id], key=lambda item: str(item["path"]))
        sources.append(
            {
                "source_id": source_id,
                "relative_root": SOURCE_ROOTS[source_id],
                "artifacts": artifacts,
                "file_count": len(artifacts),
                "bytes": sum(int(item["bytes"]) for item in artifacts),
            }
        )
    sources.extend(({"source_id": "finqa"}, {"source_id": "tat-qa"}))
    receipt = {
        "schema_version": "molor.business-source-snapshot.v1",
        "snapshot_root_label": root.name,
        "license_or_rights_metadata_used_for_admission": license_gate,
        "sources": sources,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return root, receipt_path


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_reference_build_is_deterministic_grounded_redacted_and_zero_drop(tmp_path) -> None:
    root, receipt = _build_fixture(tmp_path)
    config = BuildConfig(minimum_characters=80, maximum_characters=400, near_duplicate_distance=3)
    outputs = (tmp_path / "output-a", tmp_path / "output-b")
    manifests = [
        build_business_reference_corpus(
            snapshot_root=root,
            receipt_path=receipt,
            output_directory=output,
            config=config,
        )
        for output in outputs
    ]

    assert manifests[0] == manifests[1]
    manifest = manifests[0]
    assert manifest["artifact_accounting"]["input"] == 13
    assert manifest["artifact_accounting"]["zero_silent_drops"] is True
    assert manifest["document_accounting"]["zero_silent_drops"] is True
    assert manifest["license_or_rights_metadata_used_for_admission"] is False
    assert manifest["contains_synthetic_sft"] is False
    assert manifest["split_policy"]["family_leakage_count"] == 0
    assert manifest["document_accounting"]["statuses"]["duplicate_exact"] >= 1
    assert manifest["document_accounting"]["statuses"]["duplicate_near"] >= 1
    assert manifest["redaction_policy"]["replacement_counts"]["email"] >= 1
    assert manifest["redaction_policy"]["replacement_counts"]["phone"] >= 1
    assert manifest["redaction_policy"]["replacement_counts"]["secret_assignment"] >= 1
    assert manifest["container_member_accounting"]["zero_silent_drops"] is True
    assert manifest["container_member_accounting"]["statuses"]["selected"] == 6
    assert manifest["container_member_accounting"]["statuses"]["quarantined_unselected_table"] == 1

    emitted_leaves = {shard["leaf_id"] for shard in manifest["shards"] if shard["records"] > 0}
    assert emitted_leaves == set(LEAVES)
    all_payload = b"".join(
        (outputs[0] / shard["path"]).read_bytes() for shard in manifest["shards"]
    )
    assert b"ops@example.com" not in all_payload
    assert b"415-555-1212" not in all_payload
    assert b"abcdefghijk123456789" not in all_payload
    assert b"fixture-token-never-emitted" not in all_payload
    assert b"[REDACTED_EMAIL]" in all_payload
    assert b"[REDACTED_PHONE]" in all_payload
    assert b"[REDACTED_SECRET]" in all_payload
    for line in all_payload.splitlines():
        assert set(json.loads(line)) == {"document_id", "text", "title"}

    artifact_rows = _jsonl(outputs[0] / "provenance" / "artifacts.jsonl")
    document_rows = _jsonl(outputs[0] / "provenance" / "documents.jsonl")
    assert len(artifact_rows) == 13
    assert manifest["artifact_accounting"]["by_parser"]["pptx-ooxml-text-v1"]["emitted"] == 1
    assert any(row["status"] == "quarantined" for row in artifact_rows)
    assert all("text" not in row and "title" not in row for row in artifact_rows)
    assert all("text" not in row and "title" not in row for row in document_rows)
    families: dict[str, str] = {}
    for row in document_rows:
        prior = families.setdefault(row["family_sha256"], row["split"])
        assert prior == row["split"]

    for relative in (
        "manifest.json",
        "sha256sums.txt",
        "provenance/artifacts.jsonl",
        "provenance/documents.jsonl",
        "provenance/container-members.jsonl",
    ):
        assert (outputs[0] / relative).read_bytes() == (outputs[1] / relative).read_bytes()
    for shard in manifest["shards"]:
        assert (outputs[0] / shard["path"]).read_bytes() == (
            outputs[1] / shard["path"]
        ).read_bytes()


def test_receipt_and_artifact_pins_fail_closed_before_materialization(tmp_path) -> None:
    root, receipt_path = _build_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["license_or_rights_metadata_used_for_admission"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="license metadata"):
        load_business_receipt(receipt_path)

    root, receipt_path = _build_fixture(tmp_path / "second")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected = next(
        source for source in receipt["sources"] if source["source_id"] == "gitlab-company-handbook"
    )
    selected["artifacts"][0]["path"] = "content/handbook/security-notes.md"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked or escaping"):
        load_business_receipt(receipt_path)

    root, receipt_path = _build_fixture(tmp_path / "third")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selected = next(
        source
        for source in receipt["sources"]
        if source["source_id"] == "sba-business-guide-and-content-api"
    )
    selected["artifacts"][0]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "pin-failure-output"
    with pytest.raises(ValueError, match="hash drift"):
        build_business_reference_corpus(
            snapshot_root=root,
            receipt_path=receipt_path,
            output_directory=output,
            config=BuildConfig(minimum_characters=80, maximum_characters=400),
        )
    assert not output.exists()


def test_cli_dry_run_validates_everything_without_writing_output(tmp_path, capsys) -> None:
    root, receipt = _build_fixture(tmp_path)
    output = tmp_path / "dry-output"
    assert (
        main(
            [
                "--snapshot-root",
                str(root),
                "--receipt",
                str(receipt),
                "--output",
                str(output),
                "--minimum-characters",
                "80",
                "--maximum-characters",
                "400",
                "--dry-run",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["artifact_accounting"]["zero_silent_drops"] is True
    assert report["document_accounting"]["zero_silent_drops"] is True
    assert not output.exists()
