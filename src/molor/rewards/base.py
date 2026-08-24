"""Base interface for deterministic reward verifiers."""

from abc import ABC, abstractmethod
import hashlib
import inspect
from pathlib import Path
from typing import Any

from molor.rewards.types import EvaluationSample, VerificationResult


class Verifier(ABC):
    """A versioned, inspectable completion verifier."""

    name: str
    version: str

    @abstractmethod
    def verify(self, sample: EvaluationSample) -> VerificationResult:
        """Evaluate one sample without mutating model or experiment state."""

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable configuration used in reward fingerprints."""

        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "version": self.version,
            "implementation_sha256": self._implementation_sha256(),
        }

    def _implementation_sha256(self) -> str:
        """Hash the defining module so code edits invalidate cached reward identity."""

        module = inspect.getmodule(type(self))
        source_path = inspect.getsourcefile(module) if module is not None else None
        if source_path is not None:
            source = Path(source_path).read_bytes()
        else:
            source = inspect.getsource(type(self)).encode("utf-8")
        return hashlib.sha256(source).hexdigest()
