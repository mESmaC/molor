"""Build a source-grounded conversational embedded-systems expert corpus.

This builder consumes the prepared local ``code-corpus`` created from the user's
repositories.  Unlike ``build_local_embedded_code.py``, it does not train next-text
completion.  It turns documentation sections and named source constructs into
question/answer examples whose assistant answer is copied from the source material.

The output is local/internal only.  The source repositories have mixed and possibly
proprietary licensing; do not redistribute the generated data or adapter without a
separate per-repository review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from statistics import mean
from typing import Any, Iterable, Iterator, Sequence

from transformers import AutoTokenizer


DATASET_NAME = "embedded-knowledge-v1"
SYSTEM = (
    "You are a careful embedded-systems engineering assistant. Give concrete, technically "
    "grounded answers. Preserve exact identifiers, constants, interfaces, constraints, and "
    "target/toolchain details. Do not claim MISRA certification or invent device behavior."
)

SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
DOCUMENT_EXTENSIONS = {".md", ".txt", ".rst"}
LINKER_EXTENSIONS = {".ld", ".lds", ".icf", ".sct"}
ASSEMBLY_EXTENSIONS = {".s", ".asm"}

DOMAIN_TERMS = re.compile(
    r"\b(?:embedded|firmware|microcontroller|mcu|can(?:[- ]?fd)?|lin|sent|c166|c167|"
    r"interrupt|isr|register|peripheral|bootloader|linker|rtos|freertos|gpio|uart|"
    r"spi|i2c|i2c|adc|pwm|watchdog|memory map|toolchain|cross[- ]compil|mplab|xc16|"
    r"xc32|arm|cortex|risc[- ]?v|nxp|microchip|vehicle|control module|board support|"
    r"bsp|dma|cache|volatile|atomic|endianness|bitfield|startup|vector table)\b",
    re.IGNORECASE,
)
CORRUPTION = re.compile(r"(?:\ufffd|Ãƒ|Ã‚|Ã¢|Ã°Å¸|Ã¯Â¿Â½|ï¿½)")
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
LITERAL_SECRET = re.compile(
    r"(?im)^\s*(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)\s*"
    r"[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
)
SENSITIVE_PATH = re.compile(
    r"(?:^|/)(?:secret|secrets|credential|credentials|private[-_]?key|passwords?)(?:/|\.|$)",
    re.IGNORECASE,
)
PATH_DENY_PARTS = {
    ".git",
    "assets",
    "build",
    "debug",
    "dist",
    "generated",
    "images",
    "node_modules",
    "release",
}
DOCUMENT_DENY_NAMES = {
    "changelog.md",
    "changelog.txt",
    "history.md",
    "history.txt",
    "license.md",
    "license.txt",
    "notice.md",
    "notice.txt",
}
SECTION_DENY = re.compile(
    r"^(?:table of contents|contents|license|licensing|copyright|authors?|contributors?|"
    r"contributing|acknowledg(?:e)?ments?|revision history|change ?log)$",
    re.IGNORECASE,
)
COMMENT_NOISE = re.compile(
    r"\b(?:copyright|all rights reserved|redistribution and use|spdx-license-identifier|"
    r"generated automatically|do not edit|file name|@file)\b",
    re.IGNORECASE,
)
DECLARATION_LEAD_DENY = re.compile(r"^(?:#\s*(?:if|ifdef|ifndef|include|pragma)|extern\s+\"C\")")
CONTROL_FLOW_LEAD = re.compile(r"^(?:if|for|while|switch|return|else|do|goto|break|continue)\b")
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
RESERVED_SYMBOLS = {
    "auto",
    "bool",
    "char",
    "class",
    "const",
    "double",
    "enum",
    "float",
    "int",
    "long",
    "null",
    "short",
    "signed",
    "struct",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
}

SPARSE_TARGETS = {
    "c166-c167": 450,
    "assembly": 260,
    "linker": 360,
    "misra-oriented": 420,
    "microchip-mplab-xc": 900,
}
TASK_TARGETS = {
    "misra_note": 300,
    "linker_reference": 260,
    "assembly_routine": 180,
    "macro_contract": 700,
    "type_definition": 900,
    "documentation": 1500,
    "api_contract": 2200,
}


@dataclass(frozen=True, slots=True)
class SourceRow:
    source_split: str
    source_id: str
    source: str
    repo: str
    file_path: str
    text: str
    source_sha256: str

    @property
    def extension(self) -> str:
        path = PurePosixPath(self.file_path.replace("\\", "/"))
        return path.suffix.lower() or path.name.lower()

    @property
    def group_key(self) -> str:
        return f"{self.repo}/{self.file_path}".casefold()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*parts: str) -> int:
    return int(sha256_text("\0".join(parts))[:16], 16)


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff")
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def read_rows(paths: dict[str, Path]) -> tuple[list[SourceRow], dict[str, int]]:
    rows: list[SourceRow] = []
    counts: dict[str, int] = {}
    for split, path in paths.items():
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for source_line, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                required = {"id", "source", "repo", "file_path", "text"}
                if not required.issubset(value) or not all(
                    isinstance(value[key], str) for key in required
                ):
                    raise ValueError(f"{path}:{source_line}: invalid prepared source row")
                text = normalize_text(value["text"])
                rows.append(
                    SourceRow(
                        source_split=split,
                        source_id=value["id"],
                        source=value["source"],
                        repo=value["repo"],
                        file_path=value["file_path"].replace("\\", "/"),
                        text=text,
                        source_sha256=sha256_text(text),
                    )
                )
                count += 1
        counts[split] = count
    return rows, counts


def source_eligible(row: SourceRow) -> tuple[bool, str]:
    path = PurePosixPath(row.file_path)
    parts = {part.casefold() for part in path.parts}
    if parts & PATH_DENY_PARTS:
        return False, "denied_path"
    if SENSITIVE_PATH.search(row.file_path):
        return False, "sensitive_path"
    if not row.text.strip():
        return False, "empty"
    if "\x00" in row.text or CORRUPTION.search(row.text):
        return False, "corrupt_or_binary"
    if PRIVATE_KEY.search(row.text) or LITERAL_SECRET.search(row.text):
        return False, "possible_secret"
    return True, "eligible"


def domain_tags(row: SourceRow, passage: str = "") -> tuple[str, ...]:
    source_context = f"{row.repo}\n{row.file_path}\n{row.text[:12000]}".casefold()
    passage_context = f"{row.repo}\n{row.file_path}\n{passage}".casefold()
    tags: set[str] = {"embedded-systems"}
    if re.search(
        r"\bc1(?:66|67)\b|\bxc16x\b|\bkeil\b|\buvision\b|\b\u00b5vision\b",
        source_context,
    ):
        tags.add("c166-c167")
    if re.search(
        r"\bmicrochip\b|\bmplabx?\b|\bxc(?:8|16|32)\b|\bharmony\b|\bpic(?:24|32)?\b",
        source_context,
    ):
        tags.add("microchip-mplab-xc")
    if row.extension in LINKER_EXTENSIONS or re.search(
        r"\blinker\b|\blink script\b|\bmemory map\b", passage_context
    ):
        tags.add("linker")
    if row.extension in ASSEMBLY_EXTENSIONS or re.search(
        r"\bassembly\b|\bassembler\b|\bstartup code\b", passage_context
    ):
        tags.add("assembly")
    if "misra" in passage_context:
        tags.add("misra-oriented")
    if re.search(r"\bcan(?:[- ]?fd)?\b|\blin\b|\bsent\b", passage_context):
        tags.add("vehicle-networks")
    if re.search(
        r"\bfreertos\b|\brtos\b|\bscheduler\b|\bsemaphore\b|\bmutex\b",
        passage_context,
    ):
        tags.add("rtos-concurrency")
    if re.search(
        r"\binterrupt\b|\bisr\b|\bregister\b|\bvolatile\b|\bdma\b",
        passage_context,
    ):
        tags.add("low-level-embedded")
    if row.extension in {".c", ".h"}:
        tags.add("c")
    if row.extension in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"}:
        tags.add("c++")
    return tuple(sorted(tags))


def language_for(row: SourceRow) -> str:
    return {
        ".c": "c",
        ".h": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".hh": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".ld": "ld",
        ".lds": "ld",
        ".icf": "text",
        ".sct": "text",
        ".s": "asm",
        ".asm": "asm",
    }.get(row.extension, "text")


def clean_comment(raw: str) -> str:
    value = raw.strip()
    if value.startswith("/*"):
        value = re.sub(r"^/\*+[*!]?", "", value)
        value = re.sub(r"\*+/$", "", value)
        lines = [re.sub(r"^\s*\*\s?", "", line) for line in value.splitlines()]
    else:
        lines = [re.sub(r"^\s*//[/!]?\s?", "", line) for line in value.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return normalize_text("\n".join(lines))


def meaningful_comment(value: str, *, allow_misra: bool = False) -> bool:
    if len(value) < 45 or len(re.findall(r"[A-Za-z]{3,}", value)) < 7:
        return False
    if not allow_misra and COMMENT_NOISE.search(value):
        return False
    stripped = re.sub(r"[-=*_/\s]", "", value)
    return len(stripped) >= 30


def find_balanced(text: str, opening: int, *, limit: int = 12000) -> int | None:
    if opening < 0 or opening >= len(text) or text[opening] != "{":
        return None
    depth = 0
    quote: str | None = None
    escaped = False
    in_line_comment = False
    in_block_comment = False
    end_limit = min(len(text), opening + limit)
    index = opening
    while index < end_limit:
        char = text[index]
        next_char = text[index + 1] if index + 1 < end_limit else ""
        if in_line_comment:
            if char == "\n":
                in_line_comment = False
        elif in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                index += 1
        elif quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char == "/" and next_char == "/":
            in_line_comment = True
            index += 1
        elif char == "/" and next_char == "*":
            in_block_comment = True
            index += 1
        elif char in {'"', "'"}:
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return None


def declaration_after(text: str, offset: int) -> tuple[str, int] | None:
    index = offset
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    tail = text[index : index + 1400]
    if tail.startswith(("/*", "//")):
        return None
    if DECLARATION_LEAD_DENY.match(tail) or CONTROL_FLOW_LEAD.match(tail):
        return None
    if tail.startswith("#define") or tail.startswith("# define"):
        lines = tail.splitlines(keepends=True)
        captured: list[str] = []
        for line in lines:
            captured.append(line)
            if not line.rstrip().endswith("\\"):
                break
        declaration = "".join(captured).strip()
        return (declaration, index + len("".join(captured))) if declaration else None

    brace = tail.find("{")
    semicolon = tail.find(";")
    if semicolon < 0 and brace < 0:
        return None
    if brace >= 0 and (semicolon < 0 or brace < semicolon):
        declaration = tail[:brace].strip()
        end = index + brace
    else:
        declaration = tail[: semicolon + 1].strip()
        end = index + semicolon + 1
    if (
        not declaration
        or len(declaration) > 900
        or "\n\n" in declaration
        or "/*" in declaration
        or "//" in declaration
    ):
        return None
    if declaration.count("(") != declaration.count(")"):
        return None
    return declaration, end


def symbol_from_declaration(declaration: str) -> str | None:
    define = re.match(r"#\s*define\s+([A-Za-z_]\w*)", declaration)
    if define:
        return define.group(1)
    if re.search(r"\b[A-Za-z_]\w*\s*:\s*\d", declaration):
        return None
    class_name = re.search(r"\b(?:class|struct|enum|union)\s+([A-Za-z_]\w*)", declaration)
    if class_name and "typedef" not in declaration[: class_name.start()].split():
        candidate = class_name.group(1)
        return candidate if candidate.casefold() not in RESERVED_SYMBOLS else None
    function_pointer = re.search(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)", declaration)
    if function_pointer:
        return function_pointer.group(1)
    typedef_name = re.search(r"}\s*([A-Za-z_]\w*)\s*;$", declaration, re.DOTALL)
    if typedef_name:
        return typedef_name.group(1)
    wrapped_function = re.search(r"\)\s*([A-Za-z_]\w*)\s*\(", declaration)
    if wrapped_function:
        return wrapped_function.group(1)
    overloaded = re.search(r"\boperator\s*([^\s(]+)", declaration)
    if overloaded:
        return f"operator{overloaded.group(1)}"
    function_head = declaration.split("(", 1)[0]
    if "(" in declaration:
        candidates = re.findall(r"[A-Za-z_]\w*", function_head)
        if candidates:
            candidate = candidates[-1]
            if candidate.casefold() in RESERVED_SYMBOLS:
                return None
            if len(candidates) == 1 and candidate.isupper():
                return None
            return candidate
    left = declaration.split("=", 1)[0]
    if "," in left:
        return None
    cleaned = re.sub(r"\[[^\]]*]", "", left.rstrip(";"))
    candidates = re.findall(r"[A-Za-z_]\w*", cleaned)
    if not candidates:
        return None
    candidate = candidates[-1]
    if candidate.casefold() in RESERVED_SYMBOLS:
        return None
    return candidate if IDENTIFIER.match(candidate) else None


def make_messages(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]


def raw_candidate(
    row: SourceRow,
    *,
    task_type: str,
    question: str,
    answer: str,
    start: int,
    end: int,
    subject: str,
    transform: str,
) -> dict[str, Any] | None:
    question = normalize_text(question)
    answer = normalize_text(answer)
    if not question or not answer or CORRUPTION.search(question + answer):
        return None
    messages = make_messages(question, answer)
    canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    answer_sha256 = sha256_text(answer)
    tags = domain_tags(row, f"{subject}\n{question}\n{answer[:3000]}")
    return {
        "messages": messages,
        "example_sha256": sha256_text(canonical),
        "answer_sha256": answer_sha256,
        "source_split": row.source_split,
        "source_id": row.source_id,
        "source": row.source,
        "repo": row.repo,
        "file_path": row.file_path,
        "extension": row.extension,
        "source_content_sha256": row.source_sha256,
        "source_group": row.group_key,
        "source_line_start": line_number(row.text, max(0, start)),
        "source_line_end": line_number(row.text, min(len(row.text), max(start, end))),
        "task_type": task_type,
        "subject": subject,
        "domain_tags": tags,
        "transform": transform,
    }


@dataclass(frozen=True, slots=True)
class Heading:
    level: int
    title: str
    start: int
    content_start: int


ATX_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
SETEXT_HEADING = re.compile(r"(?m)^([^\n]+)\n([=-]{3,})[ \t]*$")


def markdown_headings(text: str) -> list[Heading]:
    headings: list[Heading] = []
    fenced = False
    fence_offsets: list[tuple[int, int]] = []
    fence_start = 0
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            if fenced:
                fence_offsets.append((fence_start, cursor + len(line)))
            else:
                fence_start = cursor
            fenced = not fenced
        cursor += len(line)
    if fenced:
        fence_offsets.append((fence_start, len(text)))

    def in_fence(offset: int) -> bool:
        return any(start <= offset < end for start, end in fence_offsets)

    for match in ATX_HEADING.finditer(text):
        if not in_fence(match.start()):
            title = re.sub(r"\s+", " ", match.group(2)).strip()
            headings.append(Heading(len(match.group(1)), title, match.start(), match.end()))
    for match in SETEXT_HEADING.finditer(text):
        if not in_fence(match.start()):
            title = re.sub(r"\s+", " ", match.group(1)).strip()
            if title and not title.startswith(("#", "-", "*", "```")):
                level = 1 if match.group(2).startswith("=") else 2
                headings.append(Heading(level, title, match.start(), match.end()))
    return sorted(headings, key=lambda item: (item.start, item.level))


def documentation_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    path = PurePosixPath(row.file_path)
    if row.extension not in DOCUMENT_EXTENSIONS or path.name.casefold() in DOCUMENT_DENY_NAMES:
        return
    path_lower = row.file_path.casefold()
    document_like = (
        "docs/" in path_lower
        or "/doc/" in f"/{path_lower}"
        or path.name.casefold().startswith("readme")
        or row.repo == "AEVIT-Ecosystem-Programming-Documentation"
    )
    if not document_like or not DOMAIN_TERMS.search(f"{row.repo}\n{row.file_path}\n{row.text[:20000]}"):
        return
    headings = markdown_headings(row.text)
    if not headings:
        return
    document_title = headings[0].title
    stack: list[Heading] = []
    for index, heading in enumerate(headings):
        while stack and stack[-1].level >= heading.level:
            stack.pop()
        ancestors = [item.title for item in stack]
        stack.append(heading)
        if heading.level == 1 or SECTION_DENY.match(heading.title.strip()):
            continue
        # Use only the prose/code directly under this heading.  Child headings get
        # their own examples, avoiding repeated nested copies of the same passage.
        next_start = headings[index + 1].start if index + 1 < len(headings) else len(row.text)
        body = normalize_text(row.text[heading.content_start:next_start])
        body = re.sub(r"^(?:\s*[-*]\s+\[[^]]+]\([^)]*\)\s*){3,}", "", body)
        if len(body) < 90 or len(body) > 7200:
            continue
        if body.count("\n-") + body.count("\n*") > 80:
            continue
        context = " > ".join([document_title, *ancestors[1:], heading.title])
        subject = heading.title
        flavor = stable_int(row.source_id, str(heading.start), "documentation") % 4
        if flavor == 0:
            question = (
                f"In the `{row.repo}` embedded documentation, explain **{subject}**. "
                f"Use the context `{context}` and preserve exact names and constraints."
            )
        elif flavor == 1:
            question = (
                f"What should an embedded developer know about **{subject}** in `{row.repo}`? "
                f"The relevant documentation path is `{row.file_path}` under `{context}`."
            )
        elif flavor == 2:
            question = (
                f"Give me the project-specific reference for **{subject}** from `{row.repo}` "
                f"(`{row.file_path}`). Keep identifiers and target/toolchain details exact."
            )
        else:
            question = (
                f"How does `{row.repo}` document **{subject}** in the context `{context}`?"
            )
        answer = f"### {heading.title}\n\n{body}"
        candidate = raw_candidate(
            row,
            task_type="documentation",
            question=question,
            answer=answer,
            start=heading.start,
            end=next_start,
            subject=subject,
            transform="markdown_heading_to_grounded_qa_v1",
        )
        if candidate is not None:
            yield candidate


COMMENT_PATTERN = re.compile(r"/\*[*!].*?\*/|(?:(?m:^\s*///[^\n]*\n?)){2,}", re.DOTALL)


def api_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    if row.extension not in SOURCE_EXTENSIONS:
        return
    for match in COMMENT_PATTERN.finditer(row.text):
        comment = clean_comment(match.group(0))
        is_misra = "misra" in comment.casefold()
        if not meaningful_comment(comment, allow_misra=is_misra):
            continue
        result = declaration_after(row.text, match.end())
        if result is None:
            continue
        declaration, end = result
        symbol = symbol_from_declaration(declaration)
        if symbol is None or symbol.casefold() in {"if", "while", "for", "switch", "return"}:
            continue
        if re.search(r"\btypedef\s+(?:struct|enum|union)\b", declaration):
            continue
        language = language_for(row)
        if is_misra:
            task_type = "misra_note"
            question = (
                f"What MISRA-related rationale or deviation is documented for `{symbol}` in "
                f"`{row.repo}/{row.file_path}`? Quote the exact interface or definition involved."
            )
            transform = "misra_comment_and_declaration_to_qa_v1"
        elif declaration.lstrip().startswith(("#define", "# define")):
            if re.search(r"\b(?:callback|function|routine)\b|@(?:param|return|retval)\b", comment, re.I):
                continue
            task_type = "macro_contract"
            question = (
                f"What does `{symbol}` represent in `{row.repo}`, and how is it defined in "
                f"`{row.file_path}`? Preserve the documented constraints."
            )
            transform = "documented_macro_to_qa_v1"
        else:
            task_type = "api_contract"
            question = (
                f"What is the documented contract for `{symbol}` in `{row.repo}`, including its "
                f"declared interface and any parameter, return, timing, or safety constraints?"
            )
            transform = "doxygen_contract_and_declaration_to_qa_v1"
        answer = f"{comment}\n\n```{language}\n{declaration}\n```"
        candidate = raw_candidate(
            row,
            task_type=task_type,
            question=question,
            answer=answer,
            start=match.start(),
            end=end,
            subject=symbol,
            transform=transform,
        )
        if candidate is not None:
            yield candidate


PLAIN_COMMENT_PATTERN = re.compile(
    r"/\*(?![*!]).*?\*/|(?:(?m:^\s*//[^\n]*\n?)){2,}", re.DOTALL
)


def sparse_documented_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    """Recover documented contracts from legacy C and vendor code using plain comments.

    Older C166/C167 and Microchip sources often use ordinary block comments rather
    than Doxygen.  Restricting this broader association pass to those sparse source
    families keeps it useful without flooding the corpus with incidental comments.
    """

    if row.extension not in SOURCE_EXTENSIONS:
        return
    row_tags = set(domain_tags(row))
    if not row_tags & {"c166-c167", "microchip-mplab-xc"}:
        return
    for match in PLAIN_COMMENT_PATTERN.finditer(row.text):
        comment = clean_comment(match.group(0))
        if not meaningful_comment(comment) or len(comment) > 1800:
            continue
        result = declaration_after(row.text, match.end())
        if result is None:
            continue
        declaration, end = result
        symbol = symbol_from_declaration(declaration)
        if (
            symbol is None
            or symbol.casefold()
            in {"if", "for", "while", "switch", "return", "else", "do", "goto"}
            or re.search(r"\btypedef\s+(?:struct|enum|union)\b", declaration)
        ):
            continue
        is_macro = declaration.lstrip().startswith(("#define", "# define"))
        if not is_macro:
            function_head = declaration.split("(", 1)[0]
            identifiers = re.findall(r"[A-Za-z_]\w*", function_head)
            # A source call/condition has one callee identifier; a declaration has
            # at least a return type and function name.  Assignment calls are also
            # executable statements, not interfaces.
            if "(" not in declaration or "=" in function_head or len(identifiers) < 2:
                continue
        if is_macro and re.search(
            r"\b(?:callback|function|routine)\b|@(?:param|return|retval)\b", comment, re.I
        ):
            continue
        task_type = "macro_contract" if is_macro else "api_contract"
        question = (
            f"What source-documented behavior and exact definition does `{symbol}` have in the "
            f"legacy/target-specific file `{row.repo}/{row.file_path}`?"
        )
        answer = f"{comment}\n\n```{language_for(row)}\n{declaration}\n```"
        candidate = raw_candidate(
            row,
            task_type=task_type,
            question=question,
            answer=answer,
            start=match.start(),
            end=end,
            subject=symbol,
            transform="legacy_plain_comment_and_declaration_to_qa_v1",
        )
        if candidate is not None:
            yield candidate


TYPE_START = re.compile(r"\btypedef\s+(struct|enum|union)(?:\s+([A-Za-z_]\w*))?\s*\{")


def preceding_comment(text: str, offset: int, *, max_gap: int = 3) -> tuple[str, int] | None:
    prefix = text[:offset]
    block = re.search(r"(?s)(/\*.*?\*/)[ \t]*(?:\n[ \t]*){0," + str(max_gap) + r"}$", prefix)
    if block:
        value = clean_comment(block.group(1))
        if meaningful_comment(value, allow_misra="misra" in value.casefold()):
            return value, block.start(1)
    lines = re.search(r"(?m)((?:^[ \t]*//[^\n]*\n){1,12})[ \t]*$", prefix)
    if lines:
        value = clean_comment(lines.group(1))
        if meaningful_comment(value, allow_misra="misra" in value.casefold()):
            return value, lines.start(1)
    return None


def type_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    if row.extension not in SOURCE_EXTENSIONS:
        return
    for match in TYPE_START.finditer(row.text):
        opening = row.text.find("{", match.start(), match.end() + 1)
        close = find_balanced(row.text, opening, limit=9000)
        if close is None:
            continue
        suffix = re.match(r"\s*([A-Za-z_]\w*)\s*;", row.text[close : close + 200])
        if suffix is None:
            continue
        symbol = suffix.group(1) or match.group(2)
        if symbol is None:
            continue
        end = close + suffix.end()
        definition = normalize_text(row.text[match.start() : end])
        if len(definition) < 40 or len(definition) > 7200:
            continue
        leading = preceding_comment(row.text, match.start())
        comment = leading[0] if leading else ""
        start = leading[1] if leading else match.start()
        kind = match.group(1)
        question = (
            f"What is the exact `{kind}` definition and field/value layout of `{symbol}` in "
            f"`{row.repo}/{row.file_path}`? Include the source documentation when present."
        )
        answer_parts = [comment] if comment else []
        answer_parts.append(f"```{language_for(row)}\n{definition}\n```")
        candidate = raw_candidate(
            row,
            task_type="type_definition",
            question=question,
            answer="\n\n".join(answer_parts),
            start=start,
            end=end,
            subject=symbol,
            transform="named_type_definition_to_qa_v1",
        )
        if candidate is not None:
            yield candidate


def standalone_misra_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    if row.extension not in SOURCE_EXTENSIONS:
        return
    comments = re.compile(r"/\*.*?\*/|(?:(?m:^\s*//[^\n]*\n?)){1,20}", re.DOTALL)
    for match in comments.finditer(row.text):
        if "misra" not in match.group(0).casefold():
            continue
        comment = clean_comment(match.group(0))
        if not meaningful_comment(comment, allow_misra=True):
            continue
        result = declaration_after(row.text, match.end())
        declaration = result[0] if result else ""
        end = result[1] if result else match.end()
        subject = symbol_from_declaration(declaration) if declaration else None
        subject = subject or f"line {line_number(row.text, match.start())}"
        answer = comment
        if declaration:
            answer += f"\n\n```{language_for(row)}\n{declaration}\n```"
        candidate = raw_candidate(
            row,
            task_type="misra_note",
            question=(
                f"What MISRA-oriented note or deviation rationale is recorded near `{subject}` in "
                f"`{row.repo}/{row.file_path}`? Keep the scope and wording technically precise."
            ),
            answer=answer,
            start=match.start(),
            end=end,
            subject=subject,
            transform="misra_source_note_to_qa_v1",
        )
        if candidate is not None:
            yield candidate


def linker_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    if row.extension not in LINKER_EXTENSIONS:
        return
    seen_ranges: set[tuple[int, int]] = set()
    block_start = re.compile(
        r"(?m)^\s*(MEMORY|SECTIONS|PHDRS|OVERLAY|GROUP|\.\w[\w.]*\s*(?:\([^)]*\))?\s*[^\n{]*)\s*\{"
    )
    for match in block_start.finditer(row.text):
        opening = row.text.find("{", match.start(), match.end())
        end = find_balanced(row.text, opening, limit=16000)
        if end is None:
            continue
        start = match.start()
        span = (start, end)
        if span in seen_ranges:
            continue
        seen_ranges.add(span)
        block = normalize_text(row.text[start:end])
        if len(block) < 50 or len(block) > 10500:
            continue
        raw_name = match.group(1).strip()
        subject = raw_name.split()[0]
        if subject == "SECTIONS" and len(block) > 5000:
            continue
        question = (
            f"How does `{row.file_path}` in `{row.repo}` define the linker block `{raw_name}`? "
            "Give the exact memory regions, symbols, placement, alignment, and flags present in source."
        )
        answer = f"```ld\n{block}\n```"
        candidate = raw_candidate(
            row,
            task_type="linker_reference",
            question=question,
            answer=answer,
            start=start,
            end=end,
            subject=raw_name,
            transform="named_linker_block_to_qa_v1",
        )
        if candidate is not None:
            yield candidate


ASSEMBLY_LABEL = re.compile(r"(?m)^([A-Za-z_$][\w$@.]*)\s*:\s*(?:[#;].*)?$")


def assembly_candidates(row: SourceRow) -> Iterator[dict[str, Any]]:
    if row.extension not in ASSEMBLY_EXTENSIONS:
        return
    labels = [match for match in ASSEMBLY_LABEL.finditer(row.text) if not match.group(1).startswith(".L")]
    for index, match in enumerate(labels):
        symbol = match.group(1)
        if symbol.casefold() in {"text", "data", "bss", "code", "const"} or symbol.isdigit():
            continue
        end = labels[index + 1].start() if index + 1 < len(labels) else len(row.text)
        provisional = row.text[match.start() : end]
        terminator = re.search(
            rf"(?m)^\s*(?:\.end\s+{re.escape(symbol)}|\.size\s+{re.escape(symbol)}\s*,[^\n]*)\s*$",
            provisional,
        )
        if terminator is not None:
            end = match.start() + terminator.end()
        body = normalize_text(row.text[match.start() : end])
        lines = body.splitlines()
        if len(lines) < 3 or len(lines) > 100 or len(body) > 9000:
            continue
        previous = preceding_comment(row.text, match.start(), max_gap=2)
        comment = previous[0] if previous else ""
        start = previous[1] if previous else match.start()
        answer_parts = [comment] if comment else []
        answer_parts.append(f"```asm\n{body}\n```")
        candidate = raw_candidate(
            row,
            task_type="assembly_routine",
            question=(
                f"What exact assembly routine or labeled block is implemented at `{symbol}` in "
                f"`{row.repo}/{row.file_path}`? Include its source comments when available."
            ),
            answer="\n\n".join(answer_parts),
            start=start,
            end=end,
            subject=symbol,
            transform="assembly_label_block_to_qa_v1",
        )
        if candidate is not None:
            yield candidate


def token_ids(tokenizer: Any, messages: list[dict[str, str]], *, generating: bool) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generating,
        enable_thinking=False,
        return_dict=False,
        truncation=False,
    )
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError("tokenizer chat template did not return a flat token list")
    return value


def measure_candidate(tokenizer: Any, candidate: dict[str, Any], max_length: int) -> bool:
    messages = candidate["messages"]
    prompt_ids = token_ids(tokenizer, messages[:-1], generating=True)
    full_ids = token_ids(tokenizer, messages, generating=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizer produced inconsistent prompt and full-chat prefixes")
    assistant_tokens = len(full_ids) - len(prompt_ids)
    candidate["full_tokens"] = len(full_ids)
    candidate["prompt_tokens"] = len(prompt_ids)
    candidate["assistant_tokens"] = assistant_tokens
    return len(full_ids) <= max_length and 12 <= assistant_tokens <= max_length - 32


def candidate_priority(candidate: dict[str, Any]) -> tuple[int, int, int]:
    tags = set(candidate["domain_tags"])
    sparse = sum(tag in SPARSE_TARGETS for tag in tags)
    task_weights = {
        "documentation": 7,
        "misra_note": 7,
        "api_contract": 6,
        "linker_reference": 6,
        "assembly_routine": 6,
        "type_definition": 5,
        "macro_contract": 4,
    }
    return (-sparse, -task_weights.get(candidate["task_type"], 0), stable_int(candidate["example_sha256"]))


def generate_candidates(
    rows: Sequence[SourceRow], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    extractors = (
        documentation_candidates,
        api_candidates,
        sparse_documented_candidates,
        type_candidates,
        standalone_misra_candidates,
        linker_candidates,
        assembly_candidates,
    )
    for row in rows:
        eligible, reason = source_eligible(row)
        if not eligible:
            rejected[f"source:{reason}"] += 1
            continue
        before = len(candidates)
        for extractor in extractors:
            for candidate in extractor(row):
                if measure_candidate(tokenizer, candidate, max_length):
                    candidates.append(candidate)
                else:
                    rejected["candidate:token_length"] += 1
        if len(candidates) == before:
            rejected["source:no_grounded_candidate"] += 1

    # Canonical answer dedup prevents near-identical prompt templates from teaching the
    # same target repeatedly.  Prefer sparse-domain and higher-value task candidates.
    candidates.sort(key=candidate_priority)
    unique: list[dict[str, Any]] = []
    seen_answers: set[str] = set()
    seen_examples: set[str] = set()
    for candidate in candidates:
        if candidate["example_sha256"] in seen_examples:
            rejected["candidate:duplicate_example"] += 1
        elif candidate["answer_sha256"] in seen_answers:
            rejected["candidate:duplicate_answer"] += 1
        else:
            seen_examples.add(candidate["example_sha256"])
            seen_answers.add(candidate["answer_sha256"])
            unique.append(candidate)
    return unique, rejected


def split_for_group(group: str, seed: str, validation_percent: int) -> str:
    value = stable_int(seed, group) % 100
    return "validation" if value < validation_percent else "train"


def stratified_group_assignments(
    candidates: Sequence[dict[str, Any]], seed: str, validation_percent: int
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Assign whole files and ensure sparse task forms exist in validation."""

    groups = {candidate["source_group"] for candidate in candidates}
    assignments = {
        group: split_for_group(group, seed, validation_percent) for group in groups
    }
    minimum_validation_groups = {
        "assembly_routine": 2,
        "linker_reference": 2,
        "misra_note": 1,
        "documentation": 8,
        "macro_contract": 8,
        "type_definition": 8,
    }
    forced: dict[str, list[str]] = {}
    for task_type, minimum in minimum_validation_groups.items():
        task_groups = {
            candidate["source_group"]
            for candidate in candidates
            if candidate["task_type"] == task_type
        }
        current = {group for group in task_groups if assignments[group] == "validation"}
        needed = max(0, minimum - len(current))
        if not needed:
            continue
        choices = sorted(
            (group for group in task_groups if assignments[group] == "train"),
            key=lambda group: stable_int(seed, "validation-stratum", task_type, group),
        )
        moved = choices[:needed]
        for group in moved:
            assignments[group] = "validation"
        if moved:
            forced[task_type] = moved
    return assignments, forced


def select_diverse(candidates: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        return sorted(candidates, key=lambda item: stable_int(item["example_sha256"]))

    by_file: Counter[str] = Counter()
    by_repo: Counter[str] = Counter()
    by_task: Counter[str] = Counter()
    by_tag: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    repo_cap = max(100, int(limit * 0.22))

    def file_cap(candidate: dict[str, Any]) -> int:
        return {
            "documentation": 24,
            "assembly_routine": 40,
            "linker_reference": 24,
            "misra_note": 20,
        }.get(candidate["task_type"], 12)

    def can_take(candidate: dict[str, Any]) -> bool:
        return (
            candidate["example_sha256"] not in selected_ids
            and by_file[candidate["source_group"]] < file_cap(candidate)
            and by_repo[candidate["repo"]] < repo_cap
        )

    def take(candidate: dict[str, Any]) -> None:
        selected.append(candidate)
        selected_ids.add(candidate["example_sha256"])
        by_file[candidate["source_group"]] += 1
        by_repo[candidate["repo"]] += 1
        by_task[candidate["task_type"]] += 1
        for tag in candidate["domain_tags"]:
            by_tag[tag] += 1

    ordered = sorted(candidates, key=candidate_priority)

    # First retain distinct sparse-domain material.  This is selection, not duplication.
    for tag, target in SPARSE_TARGETS.items():
        for candidate in ordered:
            if len(selected) >= limit or by_tag[tag] >= min(target, limit):
                break
            if tag in candidate["domain_tags"] and can_take(candidate):
                take(candidate)

    # Then balance conversational task forms before filling by global priority.
    for task, target in TASK_TARGETS.items():
        for candidate in ordered:
            if len(selected) >= limit or by_task[task] >= min(target, limit):
                break
            if candidate["task_type"] == task and can_take(candidate):
                take(candidate)

    # Round-robin repositories so the large SDK tree cannot dominate the remainder.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in ordered:
        if candidate["example_sha256"] not in selected_ids:
            buckets[candidate["repo"]].append(candidate)
    repo_order = sorted(buckets, key=lambda repo: stable_int("repo-round-robin", repo))
    cursor = Counter()
    while len(selected) < limit:
        progressed = False
        for repo in repo_order:
            bucket = buckets[repo]
            while cursor[repo] < len(bucket):
                candidate = bucket[cursor[repo]]
                cursor[repo] += 1
                if can_take(candidate):
                    take(candidate)
                    progressed = True
                    break
            if len(selected) >= limit:
                break
        if not progressed:
            break

    # The diversity cap is a guardrail, not a reason to throw away grounded unique
    # material.  If a few SDK/project repositories legitimately contain most of the
    # available contracts, relax only the repository cap while retaining per-file
    # caps and exact deduplication.
    if len(selected) < limit:
        for candidate in ordered:
            if len(selected) >= limit:
                break
            relaxed_file_cap = max(file_cap(candidate), 30)
            if (
                candidate["example_sha256"] not in selected_ids
                and by_file[candidate["source_group"]] < relaxed_file_cap
            ):
                take(candidate)
    return sorted(selected, key=lambda item: stable_int("output-order", item["example_sha256"]))


def percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def summarize(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    full = [int(value["full_tokens"]) for value in examples]
    prompt = [int(value["prompt_tokens"]) for value in examples]
    answer = [int(value["assistant_tokens"]) for value in examples]

    def stats(values: Sequence[int]) -> dict[str, int | float]:
        return {
            "min": min(values, default=0),
            "mean": round(mean(values), 2) if values else 0,
            "p50": percentile(values, 0.5),
            "p90": percentile(values, 0.9),
            "p95": percentile(values, 0.95),
            "max": max(values, default=0),
        }

    tags: Counter[str] = Counter()
    for value in examples:
        tags.update(value["domain_tags"])
    return {
        "examples": len(examples),
        "unique_source_files": len({value["source_group"] for value in examples}),
        "repositories": dict(sorted(Counter(value["repo"] for value in examples).items())),
        "task_types": dict(sorted(Counter(value["task_type"] for value in examples).items())),
        "domain_tags": dict(sorted(tags.items())),
        "extensions": dict(sorted(Counter(value["extension"] for value in examples).items())),
        "tokens": {
            "full_chat": stats(full),
            "prompt": stats(prompt),
            "assistant_target": stats(answer),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def provenance_row(value: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "example_sha256",
        "answer_sha256",
        "source_split",
        "source_id",
        "source",
        "repo",
        "file_path",
        "extension",
        "source_content_sha256",
        "source_group",
        "source_line_start",
        "source_line_end",
        "task_type",
        "subject",
        "domain_tags",
        "transform",
        "full_tokens",
        "prompt_tokens",
        "assistant_tokens",
    )
    return {field: value[field] for field in fields}


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    model = args.model.resolve()
    source_paths = {split: source_root / f"{split}.jsonl" for split in ("train", "validation")}
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows, source_counts = read_rows(source_paths)
    duplicate_groups = len(rows) - len({row.group_key for row in rows})
    if duplicate_groups:
        raise ValueError(f"prepared corpus contains {duplicate_groups} duplicate repo/path groups")

    candidates, rejected = generate_candidates(rows, tokenizer, args.max_length)
    assignments, forced_validation_groups = stratified_group_assignments(
        candidates, args.split_seed, args.validation_percent
    )
    split_candidates = {"train": [], "validation": []}
    for candidate in candidates:
        split = assignments[candidate["source_group"]]
        split_candidates[split].append(candidate)

    selected = {
        "train": select_diverse(split_candidates["train"], args.train_examples),
        "validation": select_diverse(
            split_candidates["validation"], args.validation_examples
        ),
    }
    groups = {
        split: {value["source_group"] for value in values} for split, values in selected.items()
    }
    examples = {
        split: {value["example_sha256"] for value in values} for split, values in selected.items()
    }
    answers = {
        split: {value["answer_sha256"] for value in values} for split, values in selected.items()
    }
    if groups["train"] & groups["validation"]:
        raise AssertionError("file-group leakage between train and validation")
    if examples["train"] & examples["validation"]:
        raise AssertionError("example leakage between train and validation")
    if answers["train"] & answers["validation"]:
        raise AssertionError("answer leakage between train and validation")

    output_paths: dict[str, dict[str, Path]] = {}
    for split, values in selected.items():
        suffix = "" if split == "train" else ".validation"
        dataset_path = output_root / f"{DATASET_NAME}{suffix}.jsonl"
        provenance_path = output_root / f"{DATASET_NAME}.{split}.provenance.jsonl"
        write_jsonl(dataset_path, ({"messages": value["messages"]} for value in values))
        write_jsonl(provenance_path, (provenance_row(value) for value in values))
        output_paths[split] = {"dataset": dataset_path, "provenance": provenance_path}

    candidate_tasks = Counter(value["task_type"] for value in candidates)
    candidate_tags: Counter[str] = Counter()
    for value in candidates:
        candidate_tags.update(value["domain_tags"])
    report: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "purpose": (
            "source-grounded conversational embedded-systems knowledge and API-reference LoRA SFT"
        ),
        "license_and_distribution": (
            "Local/internal experimentation only. Inputs include locally supplied, third-party, "
            "and potentially proprietary material. Do not redistribute without per-source review."
        ),
        "source": {
            split: {
                "path": str(path),
                "sha256": file_sha256(path),
                "rows": source_counts[split],
            }
            for split, path in source_paths.items()
        },
        "model_tokenizer": str(model),
        "construction": {
            "answer_policy": (
                "assistant targets are copied/normalized from source documentation, comments, "
                "declarations, named type definitions, linker blocks, or assembly blocks; no "
                "model-authored factual explanations"
            ),
            "transforms": sorted({value["transform"] for value in candidates}),
            "dedup": "SHA-256 exact canonical message and exact assistant-answer dedup",
            "split": {
                "seed": args.split_seed,
                "validation_percent": args.validation_percent,
                "group_key": "case-folded repo/file_path; all tasks from a file stay in one split",
                "validation_strata_minimum_file_groups": {
                    "assembly_routine": 2,
                    "linker_reference": 2,
                    "misra_note": 1,
                    "documentation": 8,
                    "macro_contract": 8,
                    "type_definition": 8,
                },
                "file_groups_moved_to_validation": forced_validation_groups,
            },
            "selection": {
                "requested_train_examples": args.train_examples,
                "requested_validation_examples": args.validation_examples,
                "max_full_chat_tokens": args.max_length,
                "sparse_domain_targets": SPARSE_TARGETS,
                "task_targets": TASK_TARGETS,
                "per_file_cap": (
                    "24 documentation/linker, 40 assembly, 20 MISRA-note, 12 other rows in the "
                    "diversity pass; other rows may relax to 30 only to reach the requested cap"
                ),
                "per_repo_cap": (
                    "22% diversity pass, then relaxed only if unique grounded rows are needed "
                    "to reach the requested cap"
                ),
                "padding_policy": "never duplicate or template-pad to reach a requested row count",
            },
        },
        "candidate_pool": {
            "examples": len(candidates),
            "task_types": dict(sorted(candidate_tasks.items())),
            "domain_tags": dict(sorted(candidate_tags.items())),
            "split_examples": {key: len(value) for key, value in split_candidates.items()},
            "rejections": dict(sorted(rejected.items())),
        },
        "split_audit": {
            "duplicate_prepared_repo_path_groups": duplicate_groups,
            "selected_file_group_overlap": len(groups["train"] & groups["validation"]),
            "selected_example_overlap": len(examples["train"] & examples["validation"]),
            "selected_answer_overlap": len(answers["train"] & answers["validation"]),
        },
        "outputs": {},
        "recommended_first_recipe": {
            "expert_name": DATASET_NAME,
            "dataset": str(output_paths["train"]["dataset"]),
            "max_length": args.max_length,
            "epochs": 1,
            "learning_rate": 0.00005,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "validation": str(output_paths["validation"]["dataset"]),
        },
        "limitations": [
            "The adapter can reproduce errors already present in the local source documentation or code.",
            "Exact source grounding establishes provenance, not independent factual verification.",
            "MISRA-tagged material is MISRA-oriented source commentary, not the proprietary standard and not certification.",
            "Some API/type answers teach project-specific identifiers rather than general embedded principles.",
            "The whole-file validation holdout tests transfer to unseen files, not memorization of held-out facts.",
        ],
    }
    for split in ("train", "validation"):
        report["outputs"][split] = {
            **summarize(selected[split]),
            "dataset_path": str(output_paths[split]["dataset"]),
            "dataset_sha256": file_sha256(output_paths[split]["dataset"]),
            "provenance_path": str(output_paths[split]["provenance"]),
            "provenance_sha256": file_sha256(output_paths[split]["provenance"]),
        }
    report_path = output_root / f"{DATASET_NAME}.report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "Documents" / "datasets" / "prepared" / "code-corpus",
    )
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--model", type=Path, default=workspace / "models" / "Qwen3-1.7B")
    parser.add_argument("--train-examples", type=int, default=6000)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--split-seed", default="embedded-knowledge-file-holdout-v1")
    args = parser.parse_args()
    if args.train_examples < 1 or args.validation_examples < 1:
        parser.error("example limits must be positive")
    if args.max_length < 512:
        parser.error("max-length must be at least 512")
    if not 1 <= args.validation_percent <= 40:
        parser.error("validation-percent must be between 1 and 40")
    return args


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, ensure_ascii=False))
