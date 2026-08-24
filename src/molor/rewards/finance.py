"""Safe arithmetic and FinQA-program verification for finance completions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
import math
import re
from typing import Any

from molor.rewards.base import Verifier
from molor.rewards.types import EvaluationSample, VerificationResult, VerificationStatus


class _CompletionFormatError(ValueError):
    pass


class _ReferenceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _Answer:
    value: Decimal | str
    scale: str | None = None
    currency: str | None = None
    rounding_tolerance: Decimal = Decimal(0)


_NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_ANSWER_PATTERN = re.compile(
    rf"^(?P<prefix>US\$|\$|\u20ac|\u00a3|\u00a5|USD|EUR|GBP|JPY)?\s*"
    rf"(?P<number>\(?[+-]?{_NUMBER}\)?)\s*"
    rf"(?P<scale>%|percent(?:age)?|hundred|thousand|million|billion)?"
    rf"(?:\s+(?P<suffix>USD|US\s+dollars?|dollars?|EUR|euros?|GBP|pounds?|JPY|yen))?$",
    re.IGNORECASE,
)
_EXPRESSION_NUMBER = re.compile(_NUMBER)
_DSL_NUMBER = re.compile(
    r"^[+-]?(?:(?:\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$"
)
_FIELD_PATTERNS = {
    "calculation": re.compile(r"^\s*Calculation\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
    "answer": re.compile(r"^\s*Answer\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE),
}
_DSL_OPERATIONS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
}
_TABLE_OPERATIONS = {"table_max", "table_min", "table_sum", "table_average"}
_CURRENCY_ALIASES = {
    "$": "usd",
    "us$": "usd",
    "usd": "usd",
    "dollar": "usd",
    "dollars": "usd",
    "us dollar": "usd",
    "us dollars": "usd",
    "\u20ac": "eur",
    "eur": "eur",
    "euro": "eur",
    "euros": "eur",
    "\u00a3": "gbp",
    "gbp": "gbp",
    "pound": "gbp",
    "pounds": "gbp",
    "\u00a5": "jpy",
    "jpy": "jpy",
    "yen": "jpy",
}
_SCALE_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",
    "hundred": "hundred",
    "thousand": "thousand",
    "million": "million",
    "billion": "billion",
}


def _normalize_unit(value: object, aliases: Mapping[str, str], kind: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise _ReferenceError(f"reference {kind} must be a string")
    normalized = " ".join(value.casefold().split())
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise _ReferenceError(f"unsupported reference {kind}: {value!r}") from exc


def _decimal(value: str, *, error_type: type[ValueError]) -> Decimal:
    try:
        result = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise error_type("invalid numeric literal") from exc
    if not result.is_finite():
        raise error_type("numeric values must be finite")
    return result


def _parse_answer(
    value: object,
    *,
    error_type: type[ValueError],
    forced_scale: object = None,
    forced_currency: object = None,
) -> _Answer:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise error_type("answer must be a scalar string or number")
    if isinstance(value, float) and not math.isfinite(value):
        raise error_type("answer must be finite")
    text = str(value).strip()
    if text.casefold() in {"yes", "no"}:
        if forced_scale not in (None, "") or forced_currency not in (None, ""):
            raise error_type("a yes/no answer cannot carry a numeric unit")
        return _Answer(text.casefold())

    match = _ANSWER_PATTERN.fullmatch(text)
    if match is None:
        raise error_type("Answer must contain exactly one number and an optional supported unit")

    literal = match.group("number")
    negative_parentheses = literal.startswith("(") and literal.endswith(")")
    if literal.startswith("(") != literal.endswith(")"):
        raise error_type("unbalanced accounting parentheses in Answer")
    if negative_parentheses:
        literal = literal[1:-1]
        if literal.startswith(("+", "-")):
            raise error_type("signed accounting parentheses are ambiguous")
    numeric = _decimal(literal, error_type=error_type)
    if negative_parentheses:
        numeric = -numeric

    parsed_scale = _normalize_unit(match.group("scale"), _SCALE_ALIASES, "scale")
    supplied_scale = _normalize_unit(forced_scale, _SCALE_ALIASES, "scale")
    if parsed_scale is not None and supplied_scale is not None and parsed_scale != supplied_scale:
        raise error_type("Answer scale conflicts with the reference scale")
    scale = supplied_scale or parsed_scale

    prefix = _normalize_unit(match.group("prefix"), _CURRENCY_ALIASES, "currency")
    suffix = _normalize_unit(match.group("suffix"), _CURRENCY_ALIASES, "currency")
    supplied_currency = _normalize_unit(forced_currency, _CURRENCY_ALIASES, "currency")
    currencies = {item for item in (prefix, suffix, supplied_currency) if item is not None}
    if len(currencies) > 1:
        raise error_type("Answer contains conflicting currencies")
    currency = next(iter(currencies), None)

    # FinQA programs return a ratio while their answer is conventionally rendered as 7.61%.
    # Other scales label the source table's magnitude; they do not rescale program operands.
    tolerance = Decimal(0)
    unwrapped_literal = literal.lstrip("+-")
    if "." in unwrapped_literal or "e" in unwrapped_literal.casefold():
        tolerance = Decimal(1).scaleb(numeric.copy_abs().as_tuple().exponent) / Decimal(2)
    if scale == "percent":
        if tolerance == 0:
            tolerance = Decimal("0.5")
        tolerance /= Decimal(100)
        numeric /= Decimal(100)
    return _Answer(numeric, scale=scale, currency=currency, rounding_tolerance=tolerance)


class _ExpressionParser:
    """A bounded recursive-descent parser for numeric arithmetic; it never calls eval."""

    def __init__(self, text: str, *, max_tokens: int, max_abs: Decimal) -> None:
        self.max_tokens = max_tokens
        self.max_abs = max_abs
        self.currency: str | None = None
        self.tokens = self._tokenize(text)
        self.index = 0

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        index = 0
        while index < len(text):
            if text[index].isspace():
                index += 1
                continue
            match = _EXPRESSION_NUMBER.match(text, index)
            if match is not None:
                tokens.append(match.group(0))
                index = match.end()
            elif text.startswith("**", index):
                tokens.append("**")
                index += 2
            elif text[index] in "+-*/()[]%":
                tokens.append(text[index])
                index += 1
            elif text[index] in "$\u20ac\u00a3\u00a5":
                currency = _CURRENCY_ALIASES[text[index]]
                if self.currency not in {None, currency}:
                    raise _CompletionFormatError("calculation mixes currencies")
                self.currency = currency
                tokens.append("currency")
                index += 1
            else:
                raise _CompletionFormatError(f"unsupported calculation token at offset {index}")
            if len(tokens) > self.max_tokens:
                raise _CompletionFormatError("calculation exceeds the token limit")
        if not tokens:
            raise _CompletionFormatError("Calculation is empty")
        return tokens

    def parse(self) -> Decimal:
        with localcontext() as context:
            context.prec = 50
            result = self._sum()
        if self.index != len(self.tokens):
            raise _CompletionFormatError(f"unexpected calculation token: {self.tokens[self.index]!r}")
        return self._checked(result)

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise _CompletionFormatError("calculation ended unexpectedly")
        self.index += 1
        return token

    def _sum(self) -> Decimal:
        value = self._product()
        while self._peek() in {"+", "-"}:
            operator = self._take()
            operand = self._product()
            value = value + operand if operator == "+" else value - operand
            value = self._checked(value)
        return value

    def _product(self) -> Decimal:
        value = self._power()
        while self._peek() in {"*", "/"}:
            operator = self._take()
            operand = self._power()
            if operator == "/" and operand == 0:
                raise _CompletionFormatError("division by zero")
            value = value * operand if operator == "*" else value / operand
            value = self._checked(value)
        return value

    def _power(self) -> Decimal:
        value = self._unary()
        if self._peek() == "**":
            self._take()
            exponent = self._unary()
            if exponent != exponent.to_integral_value() or abs(exponent) > 12:
                raise _CompletionFormatError("exponents must be integers between -12 and 12")
            try:
                value = value ** int(exponent)
            except (InvalidOperation, ZeroDivisionError) as exc:
                raise _CompletionFormatError("invalid exponentiation") from exc
        return self._checked(value)

    def _unary(self) -> Decimal:
        if self._peek() in {"+", "-"}:
            operator = self._take()
            value = self._unary()
            return value if operator == "+" else -value
        return self._primary()

    def _primary(self) -> Decimal:
        token = self._take()
        if token == "currency":
            token = self._take()
        if token in {"(", "["}:
            value = self._sum()
            expected = ")" if token == "(" else "]"
            if self._take() != expected:
                raise _CompletionFormatError(f"expected {expected!r}")
        elif _EXPRESSION_NUMBER.fullmatch(token) is not None:
            value = _decimal(token, error_type=_CompletionFormatError)
        else:
            raise _CompletionFormatError(f"expected a number, got {token!r}")
        if self._peek() == "%":
            self._take()
            value /= Decimal(100)
        return self._checked(value)

    def _checked(self, value: Decimal) -> Decimal:
        if not value.is_finite() or abs(value) > self.max_abs:
            raise _CompletionFormatError("calculation result exceeds the numeric bound")
        return value


class _DslExecutor:
    def __init__(
        self,
        *,
        max_steps: int,
        max_abs: Decimal,
        table: object,
    ) -> None:
        self.max_steps = max_steps
        self.max_abs = max_abs
        self.table = table

    def execute(self, text: str) -> Decimal | str:
        results: list[Decimal | str] = []
        index = 0
        while index < len(text):
            while index < len(text) and text[index].isspace():
                index += 1
            head = re.match(r"([a-z_]+)\s*\(", text[index:], re.IGNORECASE)
            if head is None:
                raise _CompletionFormatError(f"expected a FinQA operation at offset {index}")
            operation = head.group(1).casefold()
            if operation not in _DSL_OPERATIONS:
                raise _CompletionFormatError(f"unsupported FinQA operation: {operation!r}")
            body_start = index + head.end()
            body_end = text.find(")", body_start)
            if body_end < 0 or "(" in text[body_start:body_end]:
                raise _CompletionFormatError("nested or unterminated FinQA operation")
            arguments = [item.strip() for item in text[body_start:body_end].split(",")]
            if len(arguments) != 2 or not all(arguments):
                raise _CompletionFormatError("FinQA operations require exactly two arguments")
            if len(results) >= self.max_steps:
                raise _CompletionFormatError("FinQA program exceeds the step limit")
            results.append(self._apply(operation, arguments[0], arguments[1], results))
            index = body_end + 1
            while index < len(text) and text[index].isspace():
                index += 1
            if index == len(text):
                break
            if text[index] != ",":
                raise _CompletionFormatError(f"unexpected FinQA token at offset {index}")
            index += 1
            if index == len(text):
                raise _CompletionFormatError("FinQA program cannot end with a comma")
        if not results:
            raise _CompletionFormatError("Calculation is empty")
        return results[-1]

    def _apply(
        self,
        operation: str,
        first: str,
        second: str,
        results: Sequence[Decimal | str],
    ) -> Decimal | str:
        if operation in _TABLE_OPERATIONS:
            return self._table(operation, first, second)
        left = self._argument(first, results)
        right = self._argument(second, results)
        if not isinstance(left, Decimal) or not isinstance(right, Decimal):
            raise _CompletionFormatError("yes/no results cannot be arithmetic operands")
        with localcontext() as context:
            context.prec = 50
            if operation == "add":
                result: Decimal | str = left + right
            elif operation == "subtract":
                result = left - right
            elif operation == "multiply":
                result = left * right
            elif operation == "divide":
                if right == 0:
                    raise _CompletionFormatError("division by zero")
                result = left / right
            elif operation == "exp":
                if right != right.to_integral_value() or abs(right) > 12:
                    raise _CompletionFormatError("exponents must be integers between -12 and 12")
                try:
                    result = left ** int(right)
                except (InvalidOperation, ZeroDivisionError) as exc:
                    raise _CompletionFormatError("invalid exponentiation") from exc
            else:
                result = "yes" if left > right else "no"
        return self._checked(result)

    def _argument(
        self,
        value: str,
        results: Sequence[Decimal | str],
    ) -> Decimal | str:
        reference = re.fullmatch(r"#(\d+)", value)
        if reference is not None:
            result_index = int(reference.group(1))
            if result_index >= len(results):
                raise _CompletionFormatError("FinQA # reference must point to an earlier step")
            return results[result_index]
        if value.casefold().startswith("const_"):
            constant = value[6:]
            if constant.casefold() == "m1":
                constant = "-1"
            value = constant
        if _DSL_NUMBER.fullmatch(value) is None:
            raise _CompletionFormatError(f"invalid FinQA numeric argument: {value!r}")
        percent = value.endswith("%")
        if percent:
            value = value[:-1]
        result = _decimal(value, error_type=_CompletionFormatError)
        return result / Decimal(100) if percent else result

    def _table(self, operation: str, label: str, unused: str) -> Decimal:
        if unused.casefold() != "none":
            raise _CompletionFormatError("a FinQA table reduction's second argument must be none")
        if not re.fullmatch(r"[\w .&/'%\-]+", label, re.UNICODE):
            raise _CompletionFormatError("unsafe FinQA table-row label")
        if not isinstance(self.table, Sequence) or isinstance(self.table, (str, bytes)):
            raise _ReferenceError("metadata.table is required for FinQA table operations")
        wanted = " ".join(label.casefold().split())
        matches: list[Sequence[object]] = []
        for row in self.table:
            if (
                isinstance(row, Sequence)
                and not isinstance(row, (str, bytes))
                and len(row) >= 2
                and " ".join(str(row[0]).casefold().split()) == wanted
            ):
                matches.append(row)
        if len(matches) != 1:
            raise _ReferenceError("FinQA table-row label is missing or ambiguous")
        numbers: list[Decimal] = []
        for cell in matches[0][1:]:
            parsed = _parse_answer(cell, error_type=_ReferenceError)
            if not isinstance(parsed.value, Decimal):
                raise _ReferenceError("FinQA table row contains a non-numeric value")
            numbers.append(parsed.value)
        if not numbers:
            raise _ReferenceError("FinQA table row contains no values")
        if operation == "table_max":
            result = max(numbers)
        elif operation == "table_min":
            result = min(numbers)
        elif operation == "table_sum":
            result = sum(numbers, Decimal(0))
        else:
            result = sum(numbers, Decimal(0)) / Decimal(len(numbers))
        checked = self._checked(result)
        assert isinstance(checked, Decimal)
        return checked

    def _checked(self, value: Decimal | str) -> Decimal | str:
        if isinstance(value, Decimal) and (not value.is_finite() or abs(value) > self.max_abs):
            raise _CompletionFormatError("calculation result exceeds the numeric bound")
        return value


def _extract_field(completion: str, name: str) -> str:
    matches = _FIELD_PATTERNS[name].findall(completion)
    if len(matches) != 1 or not matches[0].strip():
        raise _CompletionFormatError(f"completion must contain exactly one {name.title()}: line")
    return matches[0].strip()


def _reference_answer(reference: object) -> _Answer:
    if isinstance(reference, Mapping):
        if "answer" not in reference:
            raise _ReferenceError("finance reference object requires an answer field")
        scale = reference.get("scale")
        unit = reference.get("unit")
        normalized_scale = _normalize_unit(scale, _SCALE_ALIASES, "scale")
        normalized_unit = _normalize_unit(unit, _SCALE_ALIASES, "unit")
        if normalized_scale is not None and normalized_unit is not None and (
            normalized_scale != normalized_unit
        ):
            raise _ReferenceError("reference scale and unit disagree")
        return _parse_answer(
            reference["answer"],
            error_type=_ReferenceError,
            forced_scale=normalized_scale or normalized_unit,
            forced_currency=reference.get("currency"),
        )
    return _parse_answer(reference, error_type=_ReferenceError)


def _numeric_close(
    left: Decimal | str,
    right: Decimal | str,
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
    rounding_tolerance: Decimal = Decimal(0),
) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return isinstance(left, str) and isinstance(right, str) and left == right
    tolerance = max(
        absolute_tolerance,
        abs(right) * relative_tolerance,
        rounding_tolerance,
    )
    return abs(left - right) <= tolerance


def _calculation_matches_answer(
    computed: Decimal | str,
    stated: _Answer,
    *,
    absolute_tolerance: Decimal,
    relative_tolerance: Decimal,
) -> tuple[bool, str | None]:
    if _numeric_close(
        computed,
        stated.value,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        rounding_tolerance=stated.rounding_tolerance,
    ):
        return True, "ratio" if stated.scale == "percent" else None
    if stated.scale == "percent" and isinstance(stated.value, Decimal):
        # TAT-QA derivations sometimes operate in percentage points (2.9 - 2.7 = 0.2),
        # whereas FinQA division programs return a ratio (26 / 95 = 0.27368 -> 27.4%).
        display_value = stated.value * Decimal(100)
        if _numeric_close(
            computed,
            display_value,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            rounding_tolerance=stated.rounding_tolerance * Decimal(100),
        ):
            return True, "percentage_points"
    return False, None


class FinanceCalculationVerifier(Verifier):
    """Verify ``Calculation:`` / ``Answer:`` finance output without executing Python.

    ``sample.reference`` is either a scalar answer or an object containing ``answer`` plus
    optional ``scale``/``unit`` and ``currency`` fields. FinQA table reductions additionally
    consume the original two-dimensional table from ``sample.metadata["table"]``.
    """

    def __init__(
        self,
        *,
        name: str = "finance_calculation",
        version: str = "1",
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        require_reference: bool = True,
        max_calculation_chars: int = 4096,
        max_tokens: int = 256,
        max_steps: int = 32,
        max_abs: float = 1e50,
    ) -> None:
        for label, value in (
            ("absolute_tolerance", absolute_tolerance),
            ("relative_tolerance", relative_tolerance),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
        if not isinstance(require_reference, bool):
            raise TypeError("require_reference must be boolean")
        if min(max_calculation_chars, max_tokens, max_steps) < 1:
            raise ValueError("finance verifier limits must be positive")
        if not math.isfinite(max_abs) or max_abs <= 0:
            raise ValueError("max_abs must be finite and positive")
        self.name = name
        self.version = version
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.require_reference = require_reference
        self.max_calculation_chars = max_calculation_chars
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_abs = max_abs
        self._absolute_tolerance = Decimal(str(absolute_tolerance))
        self._relative_tolerance = Decimal(str(relative_tolerance))
        self._max_abs = Decimal(str(max_abs))

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        try:
            calculation_text = _extract_field(sample.completion, "calculation")
            answer_text = _extract_field(sample.completion, "answer")
            if len(calculation_text) > self.max_calculation_chars:
                raise _CompletionFormatError("Calculation exceeds max_calculation_chars")
            stated = _parse_answer(answer_text, error_type=_CompletionFormatError)
            computed = self._execute(calculation_text, sample.metadata.get("table"))
        except _ReferenceError as exc:
            return VerificationResult.unscorable(
                self.name, VerificationStatus.INVALID_REFERENCE, str(exc)
            )
        except (ArithmeticError, _CompletionFormatError) as exc:
            return VerificationResult(
                verifier=self.name,
                status=VerificationStatus.FORMAT_FAIL,
                score=0.0,
                passed=False,
                error=str(exc),
            )

        calculation_matches, percent_convention = _calculation_matches_answer(
            computed,
            stated,
            absolute_tolerance=self._absolute_tolerance,
            relative_tolerance=self._relative_tolerance,
        )
        expected: _Answer | None = None
        if sample.reference is None:
            if self.require_reference:
                return VerificationResult.unscorable(
                    self.name,
                    VerificationStatus.INVALID_REFERENCE,
                    "finance verifier requires a reference answer",
                )
        else:
            try:
                expected = _reference_answer(sample.reference)
            except _ReferenceError as exc:
                return VerificationResult.unscorable(
                    self.name, VerificationStatus.INVALID_REFERENCE, str(exc)
                )

        answer_matches = expected is None or _numeric_close(
            stated.value,
            expected.value,
            absolute_tolerance=self._absolute_tolerance,
            relative_tolerance=self._relative_tolerance,
            rounding_tolerance=max(stated.rounding_tolerance, expected.rounding_tolerance),
        )
        unit_matches = expected is None or (
            (expected.scale is None or stated.scale == expected.scale)
            and (expected.currency is None or stated.currency == expected.currency)
        )
        passed = calculation_matches and answer_matches and unit_matches
        if expected is None:
            score = float(calculation_matches)
        else:
            score = (
                0.5 * float(calculation_matches)
                + 0.4 * float(answer_matches)
                + 0.1 * float(unit_matches)
            )
        details: dict[str, Any] = {
            "mode": self._mode(calculation_text),
            "calculation": calculation_text[:500],
            "computed": str(computed),
            "stated_answer": str(stated.value),
            "stated_scale": stated.scale,
            "stated_currency": stated.currency,
            "calculation_matches_answer": calculation_matches,
            "percent_convention": percent_convention,
            "answer_matches_reference": answer_matches,
            "unit_matches_reference": unit_matches,
        }
        if expected is not None:
            details.update(
                {
                    "expected_answer": str(expected.value),
                    "expected_scale": expected.scale,
                    "expected_currency": expected.currency,
                }
            )
        return VerificationResult(
            verifier=self.name,
            status=VerificationStatus.PASS if passed else VerificationStatus.WRONG,
            score=score,
            passed=passed,
            details=details,
        )

    def _execute(self, calculation: str, table: object) -> Decimal | str:
        if self._mode(calculation) == "finqa_dsl":
            return _DslExecutor(
                max_steps=self.max_steps,
                max_abs=self._max_abs,
                table=table,
            ).execute(calculation)
        return _ExpressionParser(
            calculation,
            max_tokens=self.max_tokens,
            max_abs=self._max_abs,
        ).parse()

    @staticmethod
    def _mode(calculation: str) -> str:
        head = re.match(r"\s*([a-z_]+)\s*\(", calculation, re.IGNORECASE)
        return "finqa_dsl" if head and head.group(1).casefold() in _DSL_OPERATIONS else "expression"

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "require_reference": self.require_reference,
            "max_calculation_chars": self.max_calculation_chars,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_abs": self.max_abs,
            "finqa_operations": sorted(_DSL_OPERATIONS),
        }
