"""Protocol for delegating executable verification to an operator-managed sandbox."""

from __future__ import annotations

import json
import math
import subprocess
from typing import Any, Sequence

from molor.rewards.base import Verifier
from molor.rewards.types import EvaluationSample, VerificationResult, VerificationStatus


class ExternalCommandVerifier(Verifier):
    """Send a sample over stdin to a fixed command and parse one JSON result.

    This class does not itself create a security boundary. The configured command must launch an
    appropriate container, VM, or other sandbox before evaluating untrusted model-generated code.
    The command is operator configuration and is never derived from the model completion.
    """

    protocol = "molor.verifier.v1"

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str,
        version: str,
        runner_fingerprint: str,
        timeout_seconds: float = 30.0,
        max_output_chars: int = 100_000,
    ) -> None:
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("external verifier command must contain non-empty string arguments")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        if not runner_fingerprint:
            raise ValueError("runner_fingerprint must not be empty")
        self.command = tuple(command)
        self.name = name
        self.version = version
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.runner_fingerprint = runner_fingerprint

    def verify(self, sample: EvaluationSample) -> VerificationResult:
        payload = json.dumps(
            {
                "protocol": self.protocol,
                "request_id": sample.sample_id,
                "sample": sample.to_dict(),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        try:
            completed = subprocess.run(
                self.command,
                input=payload,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INFRA_ERROR,
                f"external verifier exceeded {self.timeout_seconds:g} seconds",
            )

        if completed.returncode != 0:
            stderr = completed.stderr[-2_000:].strip()
            return VerificationResult.unscorable(
                self.name,
                VerificationStatus.INFRA_ERROR,
                f"external verifier exited with {completed.returncode}: {stderr or 'no stderr'}",
            )
        if len(completed.stdout) > self.max_output_chars:
            raise ValueError("external verifier response exceeded max_output_chars")

        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise TypeError("external verifier response must be a JSON object")
        if response.get("protocol") != self.protocol:
            raise ValueError("external verifier returned the wrong protocol")
        if response.get("request_id") != sample.sample_id:
            raise ValueError("external verifier returned the wrong request_id")
        try:
            status = VerificationStatus(response.get("status"))
        except (TypeError, ValueError) as exc:
            raise ValueError("external verifier returned an invalid status") from exc
        score = response.get("score")
        passed = response.get("passed")
        details = response.get("details", {})
        error = response.get("error")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))):
            raise TypeError("external verifier score must be numeric or null")
        if not isinstance(passed, bool):
            raise TypeError("external verifier passed must be boolean")
        if not isinstance(details, dict):
            raise TypeError("external verifier details must be an object")
        if error is not None and not isinstance(error, str):
            raise TypeError("external verifier error must be a string or null")
        return VerificationResult(
            verifier=self.name,
            status=status,
            score=float(score) if score is not None else None,
            passed=passed,
            details=details,
            error=error,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "protocol": self.protocol,
            "runner_fingerprint": self.runner_fingerprint,
            "timeout_seconds": self.timeout_seconds,
            "max_output_chars": self.max_output_chars,
        }
