"""Generation backends used to benchmark MoLoRA and comparison models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class BackendError(RuntimeError):
    """A model backend rejected or failed a benchmark request."""


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    messages: tuple[dict[str, str], ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    enable_thinking: bool
    context_token_budget: int | None
    cache_mode: str
    seed: int
    run_id: str
    case_id: str
    repetition: int


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    text: str
    elapsed_seconds: float
    time_to_first_token_seconds: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    finish_reason: str | None
    routing_decision: dict[str, object] | None
    generation_id: str | None
    provider_metadata: dict[str, object]


class BenchmarkBackend(Protocol):
    def identity(self) -> dict[str, object]: ...

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...


def _json_request(
    url: str,
    *,
    payload: Mapping[str, object] | None,
    timeout: float,
    headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    encoded = None
    method = "GET"
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"
        method = "POST"
    request = Request(url, data=encoded, headers=request_headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BackendError(f"HTTP {exc.code} from {url}: {body[:1000]}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise BackendError(f"could not reach {url}: {exc}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise BackendError(f"{url} returned invalid JSON: {body[:500]}") from exc
    if not isinstance(value, dict):
        raise BackendError(f"{url} returned a non-object JSON response")
    return value


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result >= 0 else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class TeacherApiBackend:
    """Use the already-loaded loopback teacher API and its actual router/expert rack."""

    def __init__(
        self,
        endpoint: str,
        *,
        route: str = "auto",
        route_top_k: int = 1,
        route_temperature: float = 1.0,
        confidence_threshold: float = 0.0,
        null_bias: float = 0.0,
        timeout: float = 600.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("teacher endpoint must be an HTTP(S) URL")
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("the teacher benchmark backend is intentionally loopback-only")
        if route_top_k < 1:
            raise ValueError("route_top_k must be positive")
        if not math.isfinite(route_temperature) or route_temperature <= 0:
            raise ValueError("route_temperature must be a positive finite number")
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be from zero to one")
        if not math.isfinite(null_bias):
            raise ValueError("null_bias must be finite")
        self.endpoint = endpoint.rstrip("/")
        self.route = route.strip() or "auto"
        self.route_top_k = route_top_k
        self.route_temperature = route_temperature
        self.confidence_threshold = confidence_threshold
        self.null_bias = null_bias
        self.timeout = timeout
        self._identity: dict[str, object] | None = None

    def identity(self) -> dict[str, object]:
        if self._identity is None:
            health = _json_request(
                f"{self.endpoint}/health",
                payload=None,
                timeout=min(self.timeout, 15.0),
            )
            if health.get("status") != "ok":
                raise BackendError(f"teacher API is unhealthy: {health}")
            self._identity = {
                "backend": "teacher-api",
                "endpoint": self.endpoint,
                "protocol_version": health.get("protocol_version"),
                "model_id": health.get("model_id"),
                "experts": health.get("experts", []),
                "configured_expert_paths": health.get("configured_expert_paths", []),
                "router_checkpoint": health.get("router_checkpoint"),
                "router_loaded": health.get("router_loaded", False),
                "route_request": self.route,
                "route_top_k": self.route_top_k,
                "route_temperature": self.route_temperature,
                "confidence_threshold": self.confidence_threshold,
                "null_bias": self.null_bias,
                "applied_generation_controls": [
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "top_k",
                    "enable_thinking",
                    "context_token_budget",
                    "cache_mode",
                    "seed",
                ],
            }
        return dict(self._identity)

    def _routing(self) -> dict[str, object]:
        common: dict[str, object] = {
            "top_k": self.route_top_k,
            "temperature": self.route_temperature,
            "confidence_threshold": self.confidence_threshold,
            "null_bias": self.null_bias,
        }
        if self.route.casefold() == "auto":
            return {"mode": "auto", **common}
        weights: dict[str, float] = {}
        for item in self.route.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, raw_weight = item.rsplit("=", 1)
                try:
                    weight = float(raw_weight)
                except ValueError as exc:
                    raise ValueError(f"invalid route weight in {item!r}") from exc
            else:
                name, weight = item, 1.0
            name = name.strip()
            if name == "base/null":
                name = "base"
            if not name or weight < 0:
                raise ValueError(f"invalid manual route component {item!r}")
            weights[name] = weight
        if not weights or sum(weights.values()) <= 0:
            raise ValueError("manual route must contain a positive route weight")
        return {"mode": "manual", "manual_route": weights, **common}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.identity()
        payload: dict[str, object] = {
            "messages": list(request.messages),
            "candidates": 1,
            "persist": False,
            "seed": request.seed,
            "routing": self._routing(),
            "settings": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "top_k": request.top_k,
                "enable_thinking": request.enable_thinking,
                "context_token_budget": request.context_token_budget,
                "cache_mode": request.cache_mode,
            },
            "metadata": {
                "purpose": "benchmark",
                "benchmark_run_id": request.run_id,
                "benchmark_case_id": request.case_id,
                "benchmark_repetition": request.repetition,
            },
        }
        started = time.perf_counter()
        response = _json_request(
            f"{self.endpoint}/v1/student/generate",
            payload=payload,
            timeout=self.timeout,
        )
        wall_elapsed = time.perf_counter() - started
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise BackendError("teacher API did not return exactly one candidate")
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise BackendError("teacher API candidate is malformed")
        text = candidate.get("response")
        if not isinstance(text, str) or not text.strip():
            raise BackendError("teacher API returned an empty response")
        telemetry = candidate.get("telemetry")
        telemetry = dict(telemetry) if isinstance(telemetry, Mapping) else {}
        routing = candidate.get("routing_decision")
        routing = dict(routing) if isinstance(routing, Mapping) else None
        elapsed = _finite_or_none(telemetry.get("elapsed_seconds")) or wall_elapsed
        completion_tokens = _int_or_none(telemetry.get("completion_tokens"))
        tokens_per_second = _finite_or_none(telemetry.get("tokens_per_second"))
        if tokens_per_second is None and completion_tokens is not None and elapsed > 0:
            tokens_per_second = completion_tokens / elapsed
        return GenerationResponse(
            text=text.strip(),
            elapsed_seconds=elapsed,
            time_to_first_token_seconds=_finite_or_none(
                telemetry.get("time_to_first_token_seconds")
            ),
            prompt_tokens=_int_or_none(telemetry.get("prompt_tokens")),
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            finish_reason=(
                str(telemetry["finish_reason"])
                if telemetry.get("finish_reason") is not None
                else None
            ),
            routing_decision=routing,
            generation_id=(
                str(candidate["generation_id"])
                if candidate.get("generation_id") is not None
                else None
            ),
            provider_metadata={"wall_elapsed_seconds": wall_elapsed},
        )


class OpenAICompatibleBackend:
    """Run the same suite against LM Studio, Ollama, vLLM, or another compatible server."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        api_key_env: str = "",
        timeout: float = 600.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("OpenAI-compatible endpoint must be an HTTP(S) URL")
        if not model.strip():
            raise ValueError("an OpenAI-compatible model name is required")
        self.endpoint = endpoint.rstrip("/")
        self.model = model.strip()
        self.api_key_env = api_key_env.strip()
        self.timeout = timeout

    @property
    def _url(self) -> str:
        if self.endpoint.endswith("/chat/completions"):
            return self.endpoint
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}/chat/completions"
        return f"{self.endpoint}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        if not self.api_key_env:
            return {}
        value = os.environ.get(self.api_key_env)
        if not value:
            raise BackendError(f"API key environment variable {self.api_key_env!r} is unset")
        return {"Authorization": f"Bearer {value}"}

    def identity(self) -> dict[str, object]:
        return {
            "backend": "openai-compatible",
            "endpoint": self.endpoint,
            "model_id": self.model,
            "api_key_env": self.api_key_env or None,
            "applied_generation_controls": [
                "max_new_tokens",
                "temperature",
                "top_p",
                "seed",
            ],
            "unsupported_generation_controls": [
                "top_k",
                "enable_thinking",
                "context_token_budget",
                "cache_mode",
            ],
        }

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(request.messages),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "stream": False,
        }
        started = time.perf_counter()
        response = _json_request(
            self._url,
            payload=payload,
            timeout=self.timeout,
            headers=self._headers(),
        )
        elapsed = time.perf_counter() - started
        choices = response.get("choices")
        if not isinstance(choices, Sequence) or not choices:
            raise BackendError("OpenAI-compatible endpoint returned no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise BackendError("OpenAI-compatible choice is malformed")
        message = choice.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise BackendError("OpenAI-compatible response has no message content")
        text = str(message["content"]).strip()
        if not text:
            raise BackendError("OpenAI-compatible endpoint returned an empty response")
        usage = response.get("usage")
        usage = dict(usage) if isinstance(usage, Mapping) else {}
        completion_tokens = _int_or_none(usage.get("completion_tokens"))
        return GenerationResponse(
            text=text,
            elapsed_seconds=elapsed,
            time_to_first_token_seconds=None,
            prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
            completion_tokens=completion_tokens,
            tokens_per_second=(
                completion_tokens / elapsed
                if completion_tokens is not None and elapsed > 0
                else None
            ),
            finish_reason=(
                str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None
            ),
            routing_decision=None,
            generation_id=(str(response["id"]) if response.get("id") is not None else None),
            provider_metadata={
                "created": response.get("created"),
                "system_fingerprint": response.get("system_fingerprint"),
            },
        )


class LmStudioNativeBackend:
    """Use LM Studio's native v1 API, including explicit reasoning control and telemetry."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str,
        model_artifact_path: str = "",
        timeout: float = 600.0,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("LM Studio endpoint must be an HTTP(S) URL")
        if not model.strip():
            raise ValueError("an LM Studio model identifier is required")
        self.endpoint = endpoint.rstrip("/")
        self.model = model.strip()
        self.model_artifact_path = model_artifact_path.strip()
        self.timeout = timeout

    @property
    def _url(self) -> str:
        if self.endpoint.endswith("/api/v1/chat"):
            return self.endpoint
        if self.endpoint.endswith("/api/v1"):
            return f"{self.endpoint}/chat"
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint[:-3]}/api/v1/chat"
        return f"{self.endpoint}/api/v1/chat"

    def identity(self) -> dict[str, object]:
        return {
            "backend": "lmstudio-native",
            "endpoint": self.endpoint,
            "model_id": self.model,
            "model_artifact_path": self.model_artifact_path or None,
            "reasoning_control": "suite-enable-thinking-to-native-on-off",
            "store": False,
            "applied_generation_controls": [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "enable_thinking",
                "context_token_budget",
            ],
            "unsupported_generation_controls": ["cache_mode", "seed"],
        }

    @staticmethod
    def _prompt(messages: Sequence[Mapping[str, str]]) -> tuple[str, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str):
                raise BackendError("LM Studio benchmark messages must contain text")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                raise BackendError(
                    "LM Studio native benchmark backend supports system/user prompts only"
                )
        if len(user_parts) != 1:
            raise BackendError("LM Studio native benchmark backend requires one user prompt")
        return "\n\n".join(system_parts).strip(), user_parts[0]

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        system_prompt, prompt = self._prompt(request.messages)
        payload: dict[str, object] = {
            "model": self.model,
            "input": prompt,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_new_tokens,
            "reasoning": "on" if request.enable_thinking else "off",
            "store": False,
            "stream": False,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if request.top_k > 0:
            payload["top_k"] = request.top_k
        if request.context_token_budget is not None:
            payload["context_length"] = request.context_token_budget

        started = time.perf_counter()
        response = _json_request(self._url, payload=payload, timeout=self.timeout)
        wall_elapsed = time.perf_counter() - started
        output = response.get("output")
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise BackendError("LM Studio native endpoint returned no output items")
        parts = [
            str(item["content"])
            for item in output
            if isinstance(item, Mapping)
            and item.get("type") == "message"
            and isinstance(item.get("content"), str)
        ]
        text = "\n".join(parts).strip()
        if not text:
            raise BackendError("LM Studio native endpoint returned an empty response")
        stats = response.get("stats")
        stats = dict(stats) if isinstance(stats, Mapping) else {}
        completion_tokens = _int_or_none(stats.get("total_output_tokens"))
        elapsed = wall_elapsed
        tokens_per_second = _finite_or_none(stats.get("tokens_per_second"))
        return GenerationResponse(
            text=text,
            elapsed_seconds=elapsed,
            time_to_first_token_seconds=_finite_or_none(
                stats.get("time_to_first_token_seconds")
            ),
            prompt_tokens=_int_or_none(stats.get("input_tokens")),
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            finish_reason=None,
            routing_decision=None,
            generation_id=(
                str(response["response_id"]) if response.get("response_id") is not None else None
            ),
            provider_metadata={
                "model_instance_id": response.get("model_instance_id"),
                "reasoning_output_tokens": _int_or_none(
                    stats.get("reasoning_output_tokens")
                ),
                "wall_elapsed_seconds": wall_elapsed,
            },
        )


class OpenCodeCliBackend:
    """Run a tool-free OpenCode model/variant and grade only final text events."""

    def __init__(
        self,
        cli_path: str,
        *,
        model: str,
        variant: str,
        agent: str,
        working_directory: str,
        timeout: float = 600.0,
    ) -> None:
        raw_cli_path = cli_path.strip()
        discovered_cli = shutil.which(raw_cli_path) if Path(raw_cli_path).name == raw_cli_path else None
        self.cli_path = str(Path(discovered_cli or raw_cli_path).expanduser().resolve())
        if not Path(self.cli_path).is_file():
            raise ValueError(f"OpenCode CLI does not exist: {self.cli_path}")
        if not model.strip() or "/" not in model:
            raise ValueError("OpenCode model must use provider/model form")
        if not variant.strip():
            raise ValueError("OpenCode variant is required")
        if not agent.strip():
            raise ValueError("OpenCode agent is required")
        self.model = model.strip()
        self.variant = variant.strip()
        self.agent = agent.strip()
        self.working_directory = str(Path(working_directory).expanduser().resolve())
        if not Path(self.working_directory).is_dir():
            raise ValueError(
                f"OpenCode working directory does not exist: {self.working_directory}"
            )
        config_path = Path(self.working_directory) / "opencode.json"
        if not config_path.is_file():
            raise ValueError(
                f"isolated OpenCode exam config does not exist: {config_path}"
            )
        self.config_path = str(config_path)
        self.config_sha256 = self._sha256(config_path)
        self.cli_sha256 = self._sha256(Path(self.cli_path))
        self.timeout = timeout
        version = subprocess.run(
            [self.cli_path, "--version"],
            cwd=self.working_directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=min(timeout, 30.0),
            check=False,
        )
        self.cli_version = version.stdout.strip() or None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def identity(self) -> dict[str, object]:
        return {
            "backend": "opencode-cli",
            "model_id": self.model,
            "variant": self.variant,
            "agent": self.agent,
            "cli_version": self.cli_version,
            "cli_sha256": self.cli_sha256,
            "working_directory": self.working_directory,
            "exam_config_path": self.config_path,
            "exam_config_sha256": self.config_sha256,
            "tool_access": "denied-by-isolated-project-config",
            "reasoning_capture": "separate-json-events-final-text-only-graded",
            "applied_generation_controls": ["enable_thinking"],
            "fixed_generation_controls": {"variant": self.variant},
            "unsupported_generation_controls": [
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "context_token_budget",
                "cache_mode",
                "seed",
            ],
        }

    @staticmethod
    def _prompt(messages: Sequence[Mapping[str, str]]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user"} or not isinstance(content, str):
                raise BackendError("OpenCode benchmark backend supports text system/user prompts")
            parts.append(content)
        if not parts:
            raise BackendError("OpenCode benchmark request has no prompt")
        return "\n\n".join(parts)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not request.enable_thinking and request.case_id != "__warmup__":
            raise BackendError(
                "OpenCode max-thinking backend requires a thinking-enabled workload"
            )
        prompt = self._prompt(request.messages)
        command = [
            self.cli_path,
            "run",
            "--model",
            self.model,
            "--variant",
            self.variant,
            "--format",
            "json",
            "--thinking",
            "--agent",
            self.agent,
        ]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=self.working_directory,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise BackendError(
                f"OpenCode generation exceeded the {self.timeout:g}s timeout"
            ) from exc
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise BackendError(
                f"OpenCode exited with status {completed.returncode}: {detail}"
            )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        token_usage: dict[str, object] = {}
        finish_reason: str | None = None
        session_id: str | None = None
        tool_events: list[str] = []
        for raw_line in completed.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise BackendError(f"OpenCode emitted invalid JSON: {raw_line[:500]}") from exc
            if not isinstance(event, Mapping):
                continue
            if isinstance(event.get("sessionID"), str):
                session_id = str(event["sessionID"])
            event_type = str(event.get("type") or "")
            part = event.get("part")
            part = dict(part) if isinstance(part, Mapping) else {}
            part_type = str(part.get("type") or "")
            if event_type == "text" and isinstance(part.get("text"), str):
                text_parts.append(str(part["text"]))
            elif event_type == "reasoning" and isinstance(part.get("text"), str):
                reasoning_parts.append(str(part["text"]))
            elif event_type == "step_finish":
                finish_reason = str(part.get("reason")) if part.get("reason") else None
                if isinstance(part.get("tokens"), Mapping):
                    token_usage = dict(part["tokens"])
            if "tool" in event_type.casefold() or "tool" in part_type.casefold():
                tool_events.append(event_type or part_type)
        if tool_events:
            raise BackendError(
                "OpenCode attempted tool use in the isolated exam: " + ", ".join(tool_events)
            )
        text = "".join(text_parts).strip()
        if not text:
            raise BackendError("OpenCode returned no final text event")
        output_tokens = _int_or_none(token_usage.get("output")) or 0
        reasoning_tokens = _int_or_none(token_usage.get("reasoning")) or 0
        generated_tokens = output_tokens + reasoning_tokens
        return GenerationResponse(
            text=text,
            elapsed_seconds=elapsed,
            time_to_first_token_seconds=None,
            prompt_tokens=_int_or_none(token_usage.get("input")),
            completion_tokens=generated_tokens or None,
            tokens_per_second=(generated_tokens / elapsed if generated_tokens else None),
            finish_reason=finish_reason,
            routing_decision=None,
            generation_id=session_id,
            provider_metadata={
                "final_output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content": "\n".join(reasoning_parts),
                "tool_events": [],
                "wall_elapsed_seconds": elapsed,
            },
        )


__all__ = [
    "BackendError",
    "BenchmarkBackend",
    "GenerationRequest",
    "GenerationResponse",
    "LmStudioNativeBackend",
    "OpenCodeCliBackend",
    "OpenAICompatibleBackend",
    "TeacherApiBackend",
]
