"""A deliberately small native Qt playground for the local mixture engine."""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from molor.desktop.adapter_catalog import AdapterCatalogEntry, discover_adapters, path_key
from molor.desktop.backend import (
    ChatTurn,
    DesktopGenerationSettings,
    DesktopModelInfo,
    DesktopSession,
    DesktopStats,
    LoadOptions,
)
from molor.desktop.benchmark_panel import BenchmarkPanel
from molor.desktop.chat_view import AssistantMessageWidget, ChatTranscript
from molor.desktop.panels import AnnotationPanel, ContextLabPanel, TeacherPanel, TrainingPanel
from molor.experimental import AutoRouteDecision, AutoRouterPolicy, BASE_ROUTE
from molor.feedback import FeedbackStore, GenerationCreate


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CORPUS_EXPERTS = frozenset(
    {
        "embedded-knowledge-v1",
        "ocb-can-verified-v1",
        "software-architecture-v1",
        "verified-embedded-code-v1",
    }
)


class PromptBox(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submitted.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class LoadWorker(QObject):
    loaded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, options: LoadOptions) -> None:
        super().__init__()
        self.options = options

    @Slot()
    def run(self) -> None:
        try:
            self.loaded.emit(DesktopSession.load(self.options))
        except Exception as exc:  # Qt worker boundary; show the useful local error
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class GenerationWorker(QObject):
    delta = Signal(str)
    reasoning_delta = Signal(str)
    reasoning_state_changed = Signal(str)
    route_decided = Signal(object)
    stats = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        session: DesktopSession,
        messages: tuple[ChatTurn, ...],
        routing_policy: AutoRouterPolicy,
        settings: DesktopGenerationSettings,
        cancellation: threading.Event,
    ) -> None:
        super().__init__()
        self.session = session
        self.messages = messages
        self.routing_policy = routing_policy
        self.settings = settings
        self.cancellation = cancellation

    @Slot()
    def run(self) -> None:
        reasoning_state = "none"
        try:
            for update in self.session.stream_chat(
                self.messages,
                route=None,
                routing_policy=self.routing_policy,
                settings=self.settings,
                cancellation=self.cancellation,
            ):
                if update.routing_decision is not None:
                    self.route_decided.emit(update.routing_decision)
                if update.reasoning_text:
                    self.reasoning_delta.emit(update.reasoning_text)
                if update.reasoning_state != reasoning_state:
                    reasoning_state = update.reasoning_state
                    self.reasoning_state_changed.emit(reasoning_state)
                if update.text:
                    self.delta.emit(update.text)
                if update.stats is not None:
                    self.stats.emit(update.stats)
                if update.finished:
                    break
        except Exception as exc:  # Qt worker boundary; this is a local experiment
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


class AutoRouterCard(QFrame):
    """Compact request-time controls for the learned prompt router."""

    def __init__(self) -> None:
        super().__init__()
        self._router_route_ids: set[str] = set()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(9)
        layout.addWidget(_section_title("AUTO ROUTER"))

        self.router_status = QLabel("Router not loaded - Auto falls back to base / null.")
        self.router_status.setObjectName("muted")
        self.router_status.setWordWrap(True)
        layout.addWidget(self.router_status)

        row = QHBoxLayout()
        row.addWidget(QLabel("Route mode"))
        row.addStretch()
        self.force_route = QComboBox()
        self.force_route.addItem("Auto", None)
        self.force_route.addItem("Force base / null", BASE_ROUTE)
        self.force_route.setToolTip("Manual forcing is a debug and benchmark override")
        self.force_route.currentIndexChanged.connect(self._mode_changed)
        row.addWidget(self.force_route)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Top-k"))
        self.router_top_k = QSpinBox()
        self.router_top_k.setRange(1, 1)
        self.router_top_k.setValue(1)
        self.router_top_k.setToolTip("Number of routes selected from the router distribution")
        row.addStretch()
        row.addWidget(self.router_top_k)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Router temperature"))
        self.router_temperature = QDoubleSpinBox()
        self.router_temperature.setRange(0.05, 5.0)
        self.router_temperature.setSingleStep(0.05)
        self.router_temperature.setDecimals(2)
        self.router_temperature.setValue(1.0)
        row.addStretch()
        row.addWidget(self.router_temperature)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Confidence floor"))
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.05)
        self.confidence_threshold.setDecimals(2)
        self.confidence_threshold.setValue(0.30)
        self.confidence_threshold.setToolTip(
            "Fall back to base / null when the highest route probability is lower"
        )
        row.addStretch()
        row.addWidget(self.confidence_threshold)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Null bias"))
        self.null_bias = QDoubleSpinBox()
        self.null_bias.setRange(-10.0, 10.0)
        self.null_bias.setSingleStep(0.1)
        self.null_bias.setDecimals(2)
        self.null_bias.setValue(0.0)
        self.null_bias.setToolTip("Additive logit bias for the untouched base route")
        row.addStretch()
        row.addWidget(self.null_bias)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max routing tokens"))
        self.max_routing_tokens = QSpinBox()
        self.max_routing_tokens.setRange(16, 8192)
        self.max_routing_tokens.setSingleStep(16)
        self.max_routing_tokens.setValue(256)
        self.max_routing_tokens.setToolTip(
            "Pinned by the loaded router checkpoint's training feature contract"
        )
        self.max_routing_tokens.setEnabled(False)
        row.addStretch()
        row.addWidget(self.max_routing_tokens)
        layout.addLayout(row)

        layout.addWidget(QLabel("Eligible loaded experts"))
        self.eligible_experts = QListWidget()
        self.eligible_experts.setMaximumHeight(104)
        self.eligible_experts.setToolTip(
            "Unchecked experts remain resident but are masked out for automatic routing"
        )
        self.eligible_experts.itemChanged.connect(self._eligible_changed)
        layout.addWidget(self.eligible_experts)

        self.route_decision = QLabel("Decision: waiting for a prompt")
        self.route_decision.setObjectName("routeValue")
        self.route_decision.setWordWrap(True)
        layout.addWidget(self.route_decision)
        self.route_probabilities = QLabel("Probabilities: -")
        self.route_probabilities.setObjectName("muted")
        self.route_probabilities.setWordWrap(True)
        layout.addWidget(self.route_probabilities)
        self.route_reason = QLabel("")
        self.route_reason.setObjectName("muted")
        self.route_reason.setWordWrap(True)
        layout.addWidget(self.route_reason)
        self._mode_changed()

    def set_loaded_experts(self, expert_names: tuple[str, ...]) -> None:
        old_enabled = {
            str(self.eligible_experts.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.eligible_experts.count())
            if self.eligible_experts.item(index).checkState() == Qt.CheckState.Checked
        }
        had_experts = self.eligible_experts.count() > 0
        old_force = self.force_route.currentData()

        self.eligible_experts.blockSignals(True)
        self.eligible_experts.clear()
        for name in expert_names:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if not had_experts or name in old_enabled
                else Qt.CheckState.Unchecked
            )
            self.eligible_experts.addItem(item)
        self.eligible_experts.blockSignals(False)

        self.force_route.blockSignals(True)
        self.force_route.clear()
        self.force_route.addItem("Auto", None)
        self.force_route.addItem("Force base / null", BASE_ROUTE)
        for name in expert_names:
            self.force_route.addItem(f"Force {name}", name)
        force_index = self.force_route.findData(old_force)
        self.force_route.setCurrentIndex(force_index if force_index >= 0 else 0)
        self.force_route.blockSignals(False)
        self._refresh_expert_eligibility(default_checked=not had_experts)
        self._eligible_changed()
        self._mode_changed()

    def set_router_state(self, *, loaded: bool, checkpoint: str | None = None) -> None:
        if loaded:
            label = Path(checkpoint).name if checkpoint else "checkpoint"
            trained_tokens = 256
            route_ids: set[str] = set()
            if checkpoint:
                try:
                    payload = json.loads(
                        (Path(checkpoint) / "router_config.json").read_text(encoding="utf-8")
                    )
                    candidate = payload.get("feature_contract", {}).get("max_tokens")
                    if isinstance(candidate, int) and not isinstance(candidate, bool):
                        trained_tokens = candidate
                    raw_route_ids = payload.get("route_ids", ())
                    if isinstance(raw_route_ids, list):
                        route_ids = {
                            route_id
                            for route_id in raw_route_ids
                            if isinstance(route_id, str) and route_id != "base/null"
                        }
                except (OSError, ValueError, TypeError):
                    pass
            had_routes = bool(self._router_route_ids)
            self._router_route_ids = route_ids
            self._refresh_expert_eligibility(default_checked=not had_routes)
            self.max_routing_tokens.setValue(trained_tokens)
            self.router_status.setText(
                f"Router loaded: {label} · {trained_tokens}-token feature contract"
            )
            self._eligible_changed()
            self._mode_changed()
            return
        self._router_route_ids.clear()
        self._refresh_expert_eligibility(default_checked=False)
        self.router_status.setText("Router not loaded - Auto falls back to base / null.")
        self._eligible_changed()
        self._mode_changed()

    def _refresh_expert_eligibility(self, *, default_checked: bool) -> None:
        """Expose checkpoint routes as auto-eligible and keep all others manual-only."""

        self.eligible_experts.blockSignals(True)
        try:
            for index in range(self.eligible_experts.count()):
                item = self.eligible_experts.item(index)
                name = str(item.data(Qt.ItemDataRole.UserRole))
                routable = name in self._router_route_ids
                flags = item.flags()
                if routable:
                    item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                    if default_checked:
                        item.setCheckState(Qt.CheckState.Checked)
                    item.setText(name)
                    item.setToolTip("Available to the loaded auto-router")
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    item.setFlags(flags & ~Qt.ItemFlag.ItemIsUserCheckable)
                    item.setText(f"{name} (manual only)")
                    item.setToolTip("Resident adapter absent from this router checkpoint")
        finally:
            self.eligible_experts.blockSignals(False)

    def reset_decision(self) -> None:
        self.route_decision.setText("Decision: routing...")
        self.route_probabilities.setText("Probabilities: -")
        self.route_reason.clear()

    def show_decision(self, decision: AutoRouteDecision) -> None:
        selected = ", ".join(decision.selected_routes) or BASE_ROUTE
        route = ", ".join(
            f"{name} {weight:.0%}" for name, weight in decision.applied_route if weight > 0
        )
        probabilities = "  |  ".join(
            f"{name} {probability:.1%}" for name, probability in decision.probabilities
        )
        mode = "auto" if decision.mode == "auto" else "forced"
        self.route_decision.setText(
            f"Decision: {selected} ({mode}, confidence {decision.confidence:.1%})"
            + (f" -> {route}" if route and route != selected else "")
        )
        self.route_probabilities.setText(f"Probabilities: {probabilities or '-'}")
        self.route_reason.setText(decision.reason)

    def policy(self) -> AutoRouterPolicy:
        auto_experts = tuple(
            str(self.eligible_experts.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.eligible_experts.count())
            if self.eligible_experts.item(index).flags() & Qt.ItemFlag.ItemIsUserCheckable
        )
        enabled = tuple(
            str(self.eligible_experts.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.eligible_experts.count())
            if (
                self.eligible_experts.item(index).flags() & Qt.ItemFlag.ItemIsUserCheckable
                and self.eligible_experts.item(index).checkState() == Qt.CheckState.Checked
            )
        )
        forced = self.force_route.currentData()
        if forced is not None:
            mode = "manual"
            manual_route = {str(forced): 1.0}
        elif not enabled:
            # Empty means "all checkpoint routes" to the runtime.  In the UI,
            # no checked routable expert intentionally means base-only.
            mode = "manual"
            manual_route = {BASE_ROUTE: 1.0}
        else:
            mode = "auto"
            manual_route = {}
        # Empty is the runtime's forward-compatible spelling for every loaded
        # expert represented by the checkpoint. Use it when nothing is masked.
        policy_enabled = () if enabled == auto_experts else enabled
        return AutoRouterPolicy(
            mode=mode,
            top_k=self.router_top_k.value(),
            temperature=self.router_temperature.value(),
            confidence_threshold=self.confidence_threshold.value(),
            null_bias=self.null_bias.value(),
            max_prompt_tokens=self.max_routing_tokens.value(),
            enabled_experts=policy_enabled,
            manual_route=manual_route,
        )

    @Slot()
    def _eligible_changed(self) -> None:
        enabled_count = sum(
            bool(
                self.eligible_experts.item(index).flags() & Qt.ItemFlag.ItemIsUserCheckable
            )
            and self.eligible_experts.item(index).checkState() == Qt.CheckState.Checked
            for index in range(self.eligible_experts.count())
        )
        self.router_top_k.setMaximum(max(1, enabled_count + 1))

    @Slot()
    def _mode_changed(self) -> None:
        automatic = self.force_route.currentData() is None
        for control in (
            self.router_top_k,
            self.router_temperature,
            self.confidence_threshold,
            self.null_bias,
            self.max_routing_tokens,
            self.eligible_experts,
        ):
            control.setEnabled(automatic)


class MolorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MoLoRA Lab")
        self.resize(1220, 820)
        self.setMinimumSize(900, 640)

        self.session: DesktopSession | None = None
        self.history: list[ChatTurn] = []
        self._load_thread: QThread | None = None
        self._load_worker: LoadWorker | None = None
        self._generation_thread: QThread | None = None
        self._generation_worker: GenerationWorker | None = None
        self._generation_cancel: threading.Event | None = None
        self._assistant_buffer = ""
        self._reasoning_buffer = ""
        self._reasoning_state = "none"
        self._rendered_assistant_buffer = ""
        self._rendered_reasoning_buffer = ""
        self._rendered_reasoning_state = "none"
        self._current_assistant_message: AssistantMessageWidget | None = None
        self._generation_failed = False
        self._conversation_id = uuid.uuid4().hex
        self._last_prompt_messages: tuple[ChatTurn, ...] | None = None
        self._current_prompt_messages: tuple[ChatTurn, ...] = ()
        self._current_settings = DesktopGenerationSettings()
        self._current_routing_policy = AutoRouterPolicy(
            mode="manual",
            manual_route={BASE_ROUTE: 1.0},
        )
        self._current_routing_decision: AutoRouteDecision | None = None
        self._current_stats: DesktopStats | None = None
        self._add_candidate_to_history = True
        self._last_generation_id: str | None = None
        self.feedback_store = FeedbackStore()
        self._feedback_closed = False

        self._build_ui()
        self._markdown_timer = QTimer(self)
        self._markdown_timer.setSingleShot(True)
        self._markdown_timer.setInterval(40)
        self._markdown_timer.timeout.connect(self._flush_streamed_message)
        self._set_status("UNLOADED", "idle")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(20, 18, 20, 18)
        root_layout.setSpacing(14)
        self.setCentralWidget(root)

        header = QHBoxLayout()
        brand = QLabel("MoLoRA")
        brand.setObjectName("brand")
        subtitle = QLabel("LOCAL ADAPTER LAB")
        subtitle.setObjectName("eyebrow")
        header.addWidget(brand)
        header.addWidget(subtitle)
        header.addStretch()
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_label = QLabel()
        self.status_label.setObjectName("status")
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)
        root_layout.addLayout(header)

        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.addTab(self._build_chat_workspace(), "Chat")
        self.annotation_panel = AnnotationPanel(self.feedback_store)
        tabs.addTab(self.annotation_panel, "Annotation / Reward")
        self.teacher_panel = TeacherPanel(
            model_source=lambda: self.model_input.text().strip(),
            expert_paths_source=self._selected_expert_paths,
            router_source=lambda: self.router_input.text().strip(),
            before_gpu_job=self._prepare_gpu_job,
        )
        self.teacher_panel.process_started.connect(
            lambda: self._set_status("TEACHER API", "ready")
        )
        self.teacher_panel.process_finished.connect(
            lambda _code: self._set_status("UNLOADED", "idle")
        )
        tabs.addTab(self.teacher_panel, "Teacher RL")
        self.benchmark_panel = BenchmarkPanel(
            workspace_root=_WORKSPACE_ROOT,
            endpoint_source=self.teacher_panel.endpoint,
        )
        self.benchmark_panel.process_started.connect(
            lambda: self._set_status("BENCHMARK", "busy")
        )
        self.benchmark_panel.process_finished.connect(
            lambda _code: self._set_status("TEACHER API", "ready")
        )
        tabs.addTab(self.benchmark_panel, "Benchmarks")
        self.training_panel = TrainingPanel(
            model_source=lambda: self.model_input.text().strip(),
            before_gpu_job=self._prepare_gpu_job,
            export_positive_annotations=self._export_positive_annotations,
            export_preference_annotations=self._export_preference_annotations,
        )
        self.training_panel.process_started.connect(lambda: self._set_status("TRAINING", "busy"))
        self.training_panel.process_finished.connect(
            lambda _code: self._set_status("UNLOADED", "idle")
        )
        tabs.addTab(self.training_panel, "Expert Training")
        self.context_lab = ContextLabPanel(
            model_source=lambda: self.model_input.text().strip(),
            before_gpu_job=self._prepare_gpu_job,
        )
        self.context_lab.process_started.connect(lambda: self._set_status("CACHE LAB", "busy"))
        self.context_lab.process_finished.connect(
            lambda _code: self._set_status("UNLOADED", "idle")
        )
        tabs.addTab(self.context_lab, "Context Lab")
        self.tabs = tabs
        root_layout.addWidget(tabs, 1)

    def _build_chat_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 8, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_chat())
        splitter.setSizes([330, 850])
        layout.addWidget(splitter)
        return workspace

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(390)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(12)

        model_card = QFrame()
        model_card.setObjectName("card")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(16, 14, 16, 14)
        model_layout.setSpacing(8)
        model_layout.addWidget(_section_title("MODEL"))
        self.model_input = QLineEdit(
            os.environ.get("MOLOR_MODEL_ID")
            or _first_existing("models/Qwen3-1.7B")
            or "Qwen/Qwen3-1.7B"
        )
        self.model_input.setPlaceholderText("Model ID or local directory")
        model_layout.addWidget(self.model_input)

        model_layout.addWidget(QLabel("Router checkpoint"))
        router_row = QHBoxLayout()
        default_router = str(
            (_WORKSPACE_ROOT / "checkpoints/qwen3-1.7b-corpus-router-v3-mlp").resolve()
        )
        self.router_input = QLineEdit(os.environ.get("MOLOR_ROUTER") or default_router)
        self.router_input.setPlaceholderText("Optional router checkpoint directory")
        self.router_input.textChanged.connect(self._update_router_input_hint)
        router_browse = QPushButton("...")
        router_browse.setFixedWidth(34)
        router_browse.clicked.connect(
            lambda _checked=False: self._browse(self.router_input, "Choose router checkpoint")
        )
        router_row.addWidget(self.router_input, 1)
        router_row.addWidget(router_browse)
        model_layout.addLayout(router_row)
        self.router_input_hint = QLabel()
        self.router_input_hint.setObjectName("muted")
        self.router_input_hint.setWordWrap(True)
        model_layout.addWidget(self.router_input_hint)

        model_layout.addWidget(QLabel("Resident experts"))
        self._adapter_catalog: tuple[AdapterCatalogEntry, ...] = ()
        self._catalog_initialized = False
        env_manual = [
            os.environ.get("MOLOR_EXPERT_A", ""),
            os.environ.get("MOLOR_EXPERT_B", ""),
        ]
        if os.environ.get("MOLOR_EXPERT_PATHS"):
            env_manual.extend(os.environ["MOLOR_EXPERT_PATHS"].split(os.pathsep))
        self._manual_expert_paths = [value.strip() for value in env_manual if value.strip()]
        self.resident_experts = QListWidget()
        self.resident_experts.setMaximumHeight(164)
        self.resident_experts.setToolTip(
            "Checked adapters are loaded into the resident rack on the next model load"
        )
        self.resident_experts.itemChanged.connect(self._resident_selection_changed)
        model_layout.addWidget(self.resident_experts)
        self.resident_summary = QLabel()
        self.resident_summary.setObjectName("muted")
        model_layout.addWidget(self.resident_summary)
        adapter_buttons = QHBoxLayout()
        refresh_adapters = QPushButton("Refresh")
        refresh_adapters.setToolTip("Rescan local checkpoint folders after a training run")
        refresh_adapters.clicked.connect(self._refresh_adapter_catalog)
        add_adapter = QPushButton("Add directory")
        add_adapter.clicked.connect(self._add_manual_expert)
        adapter_buttons.addWidget(refresh_adapters)
        adapter_buttons.addWidget(add_adapter)
        model_layout.addLayout(adapter_buttons)
        self._refresh_adapter_catalog()
        self._update_router_input_hint()
        self.load_button = QPushButton("Load model + selected experts")
        self.load_button.setObjectName("primary")
        self.load_button.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_button)
        layout.addWidget(model_card)

        self.auto_router = AutoRouterCard()
        layout.addWidget(self.auto_router)

        generation_card = QFrame()
        generation_card.setObjectName("card")
        generation_layout = QVBoxLayout(generation_card)
        generation_layout.setContentsMargins(16, 14, 16, 14)
        generation_layout.setSpacing(9)
        generation_layout.addWidget(_section_title("REASONING + SAMPLING"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset"))
        self.reasoning_preset = QComboBox()
        self.reasoning_preset.addItems(("Direct", "Balanced", "Deliberate"))
        self.reasoning_preset.currentTextChanged.connect(self._apply_reasoning_preset)
        row.addStretch()
        row.addWidget(self.reasoning_preset)
        generation_layout.addLayout(row)
        self.thinking = QCheckBox("Qwen thinking mode")
        self.thinking.setChecked(False)
        self.thinking.toggled.connect(self._apply_recommended_sampling)
        generation_layout.addWidget(self.thinking)
        row = QHBoxLayout()
        row.addWidget(QLabel("Temperature"))
        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.7)
        row.addStretch()
        row.addWidget(self.temperature)
        generation_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Top-p / top-k"))
        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.01, 1.0)
        self.top_p.setSingleStep(0.05)
        self.top_p.setValue(0.8)
        self.top_k = QSpinBox()
        self.top_k.setRange(0, 500)
        self.top_k.setValue(20)
        row.addStretch()
        row.addWidget(self.top_p)
        row.addWidget(self.top_k)
        generation_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Max new tokens"))
        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 8192)
        self.max_tokens.setValue(512)
        row.addStretch()
        row.addWidget(self.max_tokens)
        generation_layout.addLayout(row)
        generation_layout.addWidget(QLabel("System instruction"))
        self.system_instruction = QPlainTextEdit("Answer directly and clearly.")
        self.system_instruction.setMaximumHeight(72)
        generation_layout.addWidget(self.system_instruction)
        layout.addWidget(generation_card)

        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 14, 16, 14)
        stats_layout.setSpacing(7)
        stats_layout.addWidget(_section_title("LIVE STATS"))
        self.stat_model = self._stat_row(stats_layout, "Model")
        self.stat_gpu = self._stat_row(stats_layout, "GPU")
        self.stat_device = self._stat_row(stats_layout, "Placement")
        self.stat_context = self._stat_row(stats_layout, "Context")
        self.stat_cache = self._stat_row(stats_layout, "KV cache")
        self.stat_dropped = self._stat_row(stats_layout, "Dropped")
        self.stat_ttft = self._stat_row(stats_layout, "TTFT")
        self.stat_tps = self._stat_row(stats_layout, "Speed")
        layout.addWidget(stats_card)
        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    @Slot()
    def _refresh_adapter_catalog(self) -> None:
        previously_checked = {
            path_key(path, _WORKSPACE_ROOT) for path in self._selected_expert_paths()
        }
        self._adapter_catalog = discover_adapters(_WORKSPACE_ROOT)
        catalog_keys = {
            path_key(entry.path, _WORKSPACE_ROOT) for entry in self._adapter_catalog
        }
        if not self._catalog_initialized:
            previously_checked.update(
                path_key(entry.path, _WORKSPACE_ROOT)
                for entry in self._adapter_catalog
                if entry.path.parent.name == "qwen3-1.7b-corpus-experts"
                and entry.path.name in _DEFAULT_CORPUS_EXPERTS
            )
            previously_checked.update(
                path_key(raw_path, _WORKSPACE_ROOT)
                for raw_path in self._manual_expert_paths
            )

        self.resident_experts.blockSignals(True)
        self.resident_experts.clear()
        for entry in self._adapter_catalog:
            key = path_key(entry.path, _WORKSPACE_ROOT)
            self._append_resident_expert(
                entry.picker_label,
                str(entry.path),
                entry.tooltip(_WORKSPACE_ROOT),
                checked=key in previously_checked,
            )
        for raw_path in self._manual_expert_paths:
            key = path_key(raw_path, _WORKSPACE_ROOT)
            if key in catalog_keys:
                continue
            path = Path(raw_path).expanduser()
            self._append_resident_expert(
                f"{path.name or raw_path}  -  manual",
                str(path.resolve()),
                str(path.resolve()),
                checked=key in previously_checked,
            )
        self.resident_experts.blockSignals(False)
        self._catalog_initialized = True
        self._resident_selection_changed()

    def _append_resident_expert(
        self,
        label: str,
        raw_path: str,
        tooltip: str,
        *,
        checked: bool,
    ) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, raw_path)
        item.setToolTip(tooltip)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.resident_experts.addItem(item)

    def _selected_expert_paths(self) -> tuple[str, ...]:
        if not hasattr(self, "resident_experts"):
            return ()
        return tuple(
            str(self.resident_experts.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.resident_experts.count())
            if self.resident_experts.item(index).checkState() == Qt.CheckState.Checked
        )

    @Slot()
    def _add_manual_expert(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Choose LoRA adapter directory")
        if not selected:
            return
        path = Path(selected).expanduser().resolve()
        if not (path / "adapter_config.json").is_file():
            QMessageBox.warning(
                self,
                "Not a LoRA adapter",
                f"No adapter_config.json was found in:\n{path}",
            )
            return
        wanted = path_key(path, _WORKSPACE_ROOT)
        for index in range(self.resident_experts.count()):
            item = self.resident_experts.item(index)
            if path_key(str(item.data(Qt.ItemDataRole.UserRole)), _WORKSPACE_ROOT) == wanted:
                item.setCheckState(Qt.CheckState.Checked)
                self.resident_experts.scrollToItem(item)
                return
        self._manual_expert_paths.append(str(path))
        self._append_resident_expert(
            f"{path.name}  -  manual",
            str(path),
            str(path),
            checked=True,
        )
        self._resident_selection_changed()

    @Slot()
    def _resident_selection_changed(self) -> None:
        selected = len(self._selected_expert_paths())
        total = self.resident_experts.count()
        self.resident_summary.setText(f"{selected} selected / {total} discovered")
        if self.session is not None:
            self.resident_summary.setText(
                f"{selected} selected / {total} discovered - reload to change residents"
            )

    @Slot()
    def _update_router_input_hint(self) -> None:
        raw_path = self.router_input.text().strip()
        path = Path(raw_path).expanduser() if raw_path else None
        if path is not None and (path / "router_config.json").is_file() and (
            path / "router.safetensors"
        ).is_file():
            self.router_input_hint.setText("Checkpoint ready; it will be loaded with the model.")
        elif path is not None and path.is_dir():
            self.router_input_hint.setText(
                "Directory is missing router_config.json or router.safetensors."
            )
        else:
            self.router_input_hint.setText(
                "Router not available yet - Auto will explicitly fall back to base / null."
            )

    def _stat_row(self, layout: QVBoxLayout, name: str) -> QLabel:
        row = QHBoxLayout()
        key = QLabel(name)
        key.setObjectName("muted")
        value = QLabel("—")
        value.setObjectName("statValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(key)
        row.addStretch()
        row.addWidget(value, 1)
        layout.addLayout(row)
        return value

    def _build_chat(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("chatPanel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        self.chat = ChatTranscript()
        self.chat.setToolTip(
            "Load a model, choose a route, and poke it until something interesting happens."
        )
        layout.addWidget(self.chat, 1)

        self.prompt = PromptBox()
        self.prompt.setPlaceholderText(
            "Message the local model…   Enter to send · Shift+Enter for line break"
        )
        self.prompt.setMinimumHeight(82)
        self.prompt.setMaximumHeight(180)
        self.prompt.submitted.connect(self._send)
        layout.addWidget(self.prompt)

        actions = QHBoxLayout()
        self.chat_hint = QLabel("No model loaded")
        self.chat_hint.setObjectName("muted")
        actions.addWidget(self.chat_hint)
        actions.addStretch()
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("primary")
        self.send_button.setEnabled(False)
        self.send_button.clicked.connect(self._send)
        actions.addWidget(self.stop_button)
        actions.addWidget(self.send_button)
        layout.addLayout(actions)

        feedback = QHBoxLayout()
        self.record_hint = QLabel("Reward buttons activate after a generation")
        self.record_hint.setObjectName("muted")
        feedback.addWidget(self.record_hint)
        feedback.addStretch()
        self.regenerate_button = QPushButton("Regenerate last prompt")
        self.regenerate_button.setEnabled(False)
        self.regenerate_button.clicked.connect(self._regenerate)
        feedback.addWidget(self.regenerate_button)
        feedback.addWidget(QLabel("Quick reward"))
        self.quick_reward_buttons: list[QPushButton] = []
        for reward in range(-2, 3):
            label = f"{reward:+d}" if reward else "0"
            button = QPushButton(label)
            button.setObjectName("quickReward")
            button.setFixedWidth(38)
            button.setEnabled(False)
            button.clicked.connect(
                lambda _checked=False, selected_reward=reward: self._quick_grade(selected_reward)
            )
            self.quick_reward_buttons.append(button)
            feedback.addWidget(button)
        layout.addLayout(feedback)
        return frame

    def _browse(self, target: QLineEdit, title: str = "Choose directory") -> None:
        directory = QFileDialog.getExistingDirectory(self, title)
        if directory:
            target.setText(str(Path(directory)))

    @Slot()
    def _load_model(self) -> None:
        if self._load_thread is not None or self._generation_thread is not None:
            return
        if (
            self.training_panel.is_running
            or self.context_lab.is_running
            or self.teacher_panel.is_running
        ):
            self.chat_hint.setText("Stop the active GPU job before loading inference")
            return
        options = LoadOptions(
            model_id=self.model_input.text(),
            expert_paths=self._selected_expert_paths(),
            router_checkpoint=self.router_input.text(),
        )
        self._set_status("LOADING MODEL", "busy")
        self.chat_hint.setText("Loading weights — first run may download them")
        self.load_button.setEnabled(False)
        self.send_button.setEnabled(False)

        thread = QThread(self)
        worker = LoadWorker(options)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._model_loaded)
        worker.failed.connect(self._model_load_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._load_thread_finished)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @Slot(object)
    def _model_loaded(self, session: DesktopSession) -> None:
        self.session = session
        self._show_model_info(session.info)
        self.auto_router.set_loaded_experts(session.info.expert_names)
        self.auto_router.set_router_state(
            loaded=session.info.router_loaded,
            checkpoint=session.info.router_checkpoint,
        )
        self._set_status("READY", "ready")
        self.chat_hint.setText(
            f"{len(session.info.expert_names)} resident experts · "
            f"router {'ready' if session.info.router_loaded else 'base fallback'} · local inference"
        )
        self._resident_selection_changed()
        self.send_button.setEnabled(True)

    @Slot(str)
    def _model_load_failed(self, message: str) -> None:
        self._set_status("LOAD FAILED", "error")
        self.chat_hint.setText(message)
        self._append_notice(message)

    @Slot()
    def _load_thread_finished(self) -> None:
        self._load_thread = None
        self._load_worker = None
        self.load_button.setEnabled(True)
        self.send_button.setEnabled(self.session is not None)

    def _show_model_info(self, info: DesktopModelInfo) -> None:
        self.stat_model.setText(info.model_id)
        self.stat_model.setToolTip(info.model_id)
        self.stat_gpu.setText(info.gpu)
        self.stat_gpu.setToolTip(info.gpu)
        self.stat_device.setText(info.device)
        limit = f"{info.context_limit:,}" if info.context_limit else "unknown"
        self.stat_context.setText(f"— / {limit}")
        self.stat_context.setToolTip(f"Model context maximum: {limit} tokens")

    @Slot()
    def _send(self) -> None:
        if self.session is None or self._generation_thread is not None:
            return
        content = self.prompt.toPlainText().strip()
        if not content:
            return

        user_turn = ChatTurn("user", content)
        self.history.append(user_turn)
        self.chat.append_plain("YOU", content, kind="user")
        self.prompt.clear()
        instruction = self.system_instruction.toPlainText().strip()
        messages = tuple(self.history)
        if instruction:
            messages = (ChatTurn("system", instruction), *messages)
        self._last_prompt_messages = messages
        self._start_generation(
            messages,
            settings=self._generation_settings(system_instruction=instruction),
            add_candidate_to_history=True,
        )

    @Slot()
    def _regenerate(self) -> None:
        if (
            self.session is None
            or self._generation_thread is not None
            or self._last_prompt_messages is None
        ):
            return
        system = next(
            (turn.content for turn in self._last_prompt_messages if turn.role == "system"),
            "",
        )
        self._start_generation(
            self._last_prompt_messages,
            settings=self._generation_settings(system_instruction=system),
            add_candidate_to_history=False,
        )

    def _generation_settings(self, *, system_instruction: str) -> DesktopGenerationSettings:
        return DesktopGenerationSettings(
            max_new_tokens=self.max_tokens.value(),
            temperature=self.temperature.value(),
            top_p=self.top_p.value(),
            top_k=self.top_k.value(),
            enable_thinking=self.thinking.isChecked(),
            reasoning_preset=self.reasoning_preset.currentText(),
            system_instruction=system_instruction,
            context_token_budget=self.context_lab.context_token_budget(),
            cache_mode=self.context_lab.cache_mode(),
        )

    def _start_generation(
        self,
        messages: tuple[ChatTurn, ...],
        *,
        settings: DesktopGenerationSettings,
        add_candidate_to_history: bool,
    ) -> None:
        assert self.session is not None
        self.prompt.setEnabled(False)
        self.send_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.load_button.setEnabled(False)
        self.regenerate_button.setEnabled(False)
        for button in self.quick_reward_buttons:
            button.setEnabled(False)
        self._set_status("GENERATING", "busy")
        self.chat_hint.setText("Streaming from the local model")
        self.stat_ttft.setText("…")
        self.stat_tps.setText("…")
        self._assistant_buffer = ""
        self._reasoning_buffer = ""
        self._reasoning_state = "none"
        self._rendered_assistant_buffer = ""
        self._rendered_reasoning_buffer = ""
        self._rendered_reasoning_state = "none"
        self._current_assistant_message = self.chat.begin_assistant()
        self._generation_failed = False
        self._current_prompt_messages = messages
        self._current_settings = settings
        self._current_routing_policy = self.auto_router.policy()
        self._current_routing_decision = None
        self.auto_router.reset_decision()
        self._current_stats = None
        self._add_candidate_to_history = add_candidate_to_history
        cancellation = threading.Event()
        self._generation_cancel = cancellation

        thread = QThread(self)
        worker = GenerationWorker(
            self.session,
            messages,
            self._current_routing_policy,
            settings,
            cancellation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.delta.connect(self._append_delta)
        worker.reasoning_delta.connect(self._append_reasoning_delta)
        worker.reasoning_state_changed.connect(self._set_reasoning_state)
        worker.route_decided.connect(self._route_decided)
        worker.stats.connect(self._update_stats)
        worker.failed.connect(self._generation_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._generation_finished)
        self._generation_thread = thread
        self._generation_worker = worker
        thread.start()

    @Slot(object)
    def _route_decided(self, decision: AutoRouteDecision) -> None:
        self._current_routing_decision = decision
        self.auto_router.show_decision(decision)

    @Slot(str)
    def _append_delta(self, text: str) -> None:
        self._assistant_buffer += text
        self._schedule_markdown_render()

    @Slot(str)
    def _append_reasoning_delta(self, text: str) -> None:
        self._reasoning_buffer += text
        self._schedule_markdown_render()

    @Slot(str)
    def _set_reasoning_state(self, state: str) -> None:
        self._reasoning_state = state
        self._schedule_markdown_render()

    def _schedule_markdown_render(self) -> None:
        if not self._markdown_timer.isActive():
            self._markdown_timer.start()

    @Slot()
    def _flush_streamed_message(self) -> None:
        message = self._current_assistant_message
        if message is None:
            return
        if self._assistant_buffer != self._rendered_assistant_buffer:
            message.set_answer_markdown(self._assistant_buffer)
            self._rendered_assistant_buffer = self._assistant_buffer
        if (
            self._reasoning_buffer != self._rendered_reasoning_buffer
            or self._reasoning_state != self._rendered_reasoning_state
        ):
            message.set_reasoning_markdown(
                self._reasoning_buffer,
                incomplete=self._reasoning_state == "incomplete",
                streaming=self._reasoning_state == "streaming",
            )
            self._rendered_reasoning_buffer = self._reasoning_buffer
            self._rendered_reasoning_state = self._reasoning_state

    @Slot(object)
    def _update_stats(self, stats: DesktopStats) -> None:
        self._current_stats = stats
        effective_limit = stats.context_token_budget or stats.context_limit
        limit = f"{effective_limit:,}" if effective_limit else "?"
        self.stat_context.setText(f"{stats.context_tokens:,} / {limit}")
        model_limit = f"{stats.context_limit:,}" if stats.context_limit else "unknown"
        self.stat_context.setToolTip(
            f"Effective budget: {limit} tokens · model maximum: {model_limit} tokens"
        )
        self.stat_cache.setText(stats.cache_mode)
        self.stat_dropped.setText(
            f"{stats.dropped_turns} turns / {stats.dropped_tokens:,} tok"
            if stats.dropped_turns or stats.dropped_tokens
            else "none"
        )
        self.stat_ttft.setText(
            f"{stats.time_to_first_token_seconds:.2f}s"
            if stats.time_to_first_token_seconds is not None
            else "—"
        )
        self.stat_tps.setText(
            f"{stats.tokens_per_second:.1f} tok/s" if stats.tokens_per_second is not None else "—"
        )
        suffix = f" · {stats.finish_reason}" if stats.finish_reason else ""
        self.chat_hint.setText(f"{stats.completion_tokens:,} tokens{suffix}")

    @Slot(str)
    def _generation_error(self, message: str) -> None:
        self._generation_failed = True
        if self._reasoning_state == "streaming":
            self._reasoning_state = "incomplete"
        self._markdown_timer.stop()
        self._flush_streamed_message()
        self._append_notice(message)
        self.chat_hint.setText(message)
        self._set_status("GENERATION FAILED", "error")

    @Slot()
    def _generation_finished(self) -> None:
        was_cancelled = bool(self._generation_cancel and self._generation_cancel.is_set())
        if self._reasoning_state == "streaming":
            self._reasoning_state = "incomplete"
        self._markdown_timer.stop()
        self._flush_streamed_message()
        if self._assistant_buffer and self._add_candidate_to_history:
            self.history.append(ChatTurn("assistant", self._assistant_buffer))
        if (
            self._assistant_buffer
            and self._current_stats is not None
            and not self._generation_failed
        ):
            self._record_completed_generation()
        self._generation_thread = None
        self._generation_worker = None
        self._generation_cancel = None
        self._current_assistant_message = None
        self.prompt.setEnabled(True)
        self.send_button.setEnabled(self.session is not None)
        self.stop_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.regenerate_button.setEnabled(self._last_prompt_messages is not None)
        if not self._generation_failed:
            self._set_status("READY", "ready")
            if was_cancelled:
                self.chat_hint.setText("Stopped")
        self.prompt.setFocus()

    def _record_completed_generation(self) -> None:
        stats = self._current_stats
        if stats is None:
            return
        settings = self._current_settings
        exact_system = next(
            (turn.content for turn in self._current_prompt_messages if turn.role == "system"),
            None,
        )
        try:
            generation_id = self.feedback_store.record_generation(
                GenerationCreate(
                    conversation_id=self._conversation_id,
                    model_id=stats.model_id,
                    messages=tuple(
                        {"role": turn.role, "content": turn.content}
                        for turn in self._current_prompt_messages
                    ),
                    response=self._assistant_buffer,
                    route=dict(stats.route),
                    settings={
                        "max_new_tokens": settings.max_new_tokens,
                        "temperature": settings.temperature,
                        "top_p": settings.top_p,
                        "top_k": settings.top_k,
                        "do_sample": settings.temperature > 0,
                        "routing_policy": self._current_routing_policy.to_dict(),
                    },
                    telemetry={
                        "device": stats.device,
                        "dtype": stats.dtype,
                        "prompt_tokens": stats.prompt_tokens,
                        "completion_tokens": stats.completion_tokens,
                        "context_tokens": stats.context_tokens,
                        "context_limit": stats.context_limit,
                        "elapsed_seconds": stats.elapsed_seconds,
                        "time_to_first_token_seconds": stats.time_to_first_token_seconds,
                        "tokens_per_second": stats.tokens_per_second,
                        "finish_reason": stats.finish_reason,
                        "reasoning_trace": self._reasoning_buffer,
                        "reasoning_trace_state": self._reasoning_state,
                        "routing_decision": (
                            self._current_routing_decision.to_dict()
                            if self._current_routing_decision is not None
                            else None
                        ),
                    },
                    system_instruction=exact_system,
                    enable_thinking=settings.enable_thinking,
                    reasoning_preset=settings.reasoning_preset,
                    context_policy={
                        "requested_token_budget": settings.context_token_budget,
                        "effective_token_budget": stats.context_token_budget,
                        "dropped_turns": stats.dropped_turns,
                        "dropped_tokens": stats.dropped_tokens,
                    },
                    cache_policy={"mode": stats.cache_mode},
                )
            )
        except Exception as exc:
            self.record_hint.setText(f"Feedback record failed: {type(exc).__name__}: {exc}")
            return
        self._last_generation_id = generation_id
        self.record_hint.setText(f"Recorded · {generation_id[:10]}")
        for button in self.quick_reward_buttons:
            button.setEnabled(True)
        self.annotation_panel.generation_recorded(generation_id)

    def _quick_grade(self, reward: int) -> None:
        if self._last_generation_id is None:
            return
        try:
            self.annotation_panel.quick_grade(self._last_generation_id, reward)
        except Exception as exc:
            self.record_hint.setText(f"Grade failed: {type(exc).__name__}: {exc}")
            return
        self.record_hint.setText(f"Saved {reward:+d} · {self._last_generation_id[:10]}")

    @Slot(str)
    def _apply_reasoning_preset(self, preset: str) -> None:
        presets = {
            "Direct": (False, "Answer directly and clearly. Do not add unnecessary preamble."),
            "Balanced": (
                True,
                "Reason carefully, then give a clear answer with only the useful explanation.",
            ),
            "Deliberate": (
                True,
                "Work through the problem deliberately. Check assumptions and the final answer.",
            ),
        }
        thinking, instruction = presets.get(preset, presets["Direct"])
        self.thinking.setChecked(thinking)
        self.system_instruction.setPlainText(instruction)
        self._apply_recommended_sampling(thinking)

    @Slot(bool)
    def _apply_recommended_sampling(self, thinking: bool) -> None:
        if thinking:
            self.temperature.setValue(0.6)
            self.top_p.setValue(0.95)
            self.top_k.setValue(20)
        else:
            self.temperature.setValue(0.7)
            self.top_p.setValue(0.8)
            self.top_k.setValue(20)

    def _prepare_gpu_job(self, label: str) -> bool:
        if (
            self.training_panel.is_running
            or self.context_lab.is_running
            or self.teacher_panel.is_running
        ):
            QMessageBox.information(self, "GPU busy", "Another experiment process is running.")
            return False
        if self._generation_thread is not None or self._load_thread is not None:
            QMessageBox.information(self, "Model busy", "Stop generation or loading first.")
            return False
        if self.session is None:
            return True
        answer = QMessageBox.question(
            self,
            "Unload inference model?",
            f"{label.capitalize()} needs the GPU. Unload the chat model and continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._unload_inference()
        return True

    def _unload_inference(self) -> None:
        session = self.session
        self.session = None
        self.send_button.setEnabled(False)
        self.load_button.setEnabled(True)
        self.auto_router.set_loaded_experts(())
        self.auto_router.set_router_state(loaded=False)
        self.chat_hint.setText("Inference model unloaded; load it again after the GPU job")
        self._set_status("UNLOADED", "idle")
        if session is not None:
            del session
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _export_positive_annotations(self, path: Path) -> int:
        result = self.feedback_store.export_training_sft_jsonl(
            path,
            min_human_reward=1,
            min_teacher_score=1,
            min_teacher_confidence=0.5,
        )
        return result.rows

    def _export_preference_annotations(self, path: Path) -> int:
        return self.feedback_store.export_preference_jsonl(path).rows

    @Slot()
    def _stop(self) -> None:
        if self._generation_cancel is not None:
            self._generation_cancel.set()
            self.stop_button.setEnabled(False)
            self.chat_hint.setText("Stopping after the current model step…")

    def _append_notice(self, text: str) -> None:
        self.chat.append_plain("RUNTIME", text, kind="runtime")

    def _set_status(self, text: str, state: str) -> None:
        colors = {"idle": "#6c7487", "busy": "#f5b84b", "ready": "#51d69b", "error": "#ff6475"}
        self.status_label.setText(text)
        self.status_dot.setStyleSheet(f"color: {colors[state]};")

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        if self._load_thread is not None and self._load_thread.isRunning():
            self.chat_hint.setText("Model is still loading; close again after it finishes")
            event.ignore()
            return
        if self._generation_thread is not None and self._generation_thread.isRunning():
            self._stop()
            if not self._generation_thread.wait(3000):
                self.chat_hint.setText("Waiting for the current model step to stop…")
                event.ignore()
                return
        for panel in (
            self.training_panel,
            self.context_lab,
            self.benchmark_panel,
            self.teacher_panel,
        ):
            if panel.is_running:
                panel.stop()
                if not panel.process.waitForFinished(3500):
                    panel.process.kill()
                    panel.process.waitForFinished(1000)
        if not self._feedback_closed:
            self.feedback_store.close()
            self._feedback_closed = True
        event.accept()


def _section_title(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("eyebrow")
    return label


def _first_existing(*candidates: str) -> str:
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = _WORKSPACE_ROOT / path
        if path.is_dir():
            return str(path.resolve())
    return ""


def _apply_theme(app: QApplication) -> None:
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QWidget { background: #0b0e14; color: #dbe1ed; }
        QMainWindow { background: #0b0e14; }
        #brand { font-size: 25px; font-weight: 700; color: #f0f3fa; }
        #panelTitle { font-size: 18px; font-weight: 700; color: #eef2fa; }
        #eyebrow { color: #77829a; font-size: 10px; font-weight: 700; letter-spacing: 1px; }
        #status { color: #a8b0c2; font-size: 10px; font-weight: 700; }
        #statusDot { font-size: 13px; }
        #muted { color: #7e879a; font-size: 9px; }
        #statValue { color: #dce3f2; font-size: 9px; }
        #routeValue { color: #91aaff; font-weight: 700; }
        #card, #chatPanel { background: #111620; border: 1px solid #20283a; border-radius: 9px; }
        QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #0c1119; border: 1px solid #293247; border-radius: 6px;
            padding: 7px; selection-background-color: #4d68c9;
        }
        QLineEdit:focus, QPlainTextEdit:focus { border-color: #6380e8; }
        QTextEdit { padding: 14px; font-size: 11pt; }
        QPushButton { background: #1b2231; border: 1px solid #2c374e; border-radius: 6px; padding: 7px 11px; }
        QPushButton:hover { background: #242e42; }
        QPushButton:disabled { color: #555e70; background: #141923; border-color: #202737; }
        QPushButton#primary { background: #5774dc; border-color: #718bef; color: white; font-weight: 700; }
        QPushButton#primary:hover { background: #6681e5; }
        QPushButton#danger { color: #ff9ba5; border-color: #633844; }
        QPushButton#preset { padding: 5px 6px; font-size: 9px; }
        QPushButton#quickReward { padding: 5px; font-size: 9px; }
        QTabWidget::pane { border: 1px solid #20283a; border-radius: 7px; top: -1px; }
        QTabBar::tab { background: #10151e; border: 1px solid #20283a; padding: 8px 15px; }
        QTabBar::tab:selected { background: #1a2232; color: #aebfff; border-bottom-color: #1a2232; }
        QTabBar::tab:hover { background: #171e2b; }
        QSlider::groove:horizontal { height: 4px; background: #293247; border-radius: 2px; }
        QSlider::sub-page:horizontal { background: #6682e6; border-radius: 2px; }
        QSlider::handle:horizontal { background: #dce4ff; width: 13px; margin: -5px 0; border-radius: 6px; }
        QScrollArea { background: transparent; }
        QScrollBar:vertical { background: #0d1119; width: 9px; }
        QScrollBar::handle:vertical { background: #2e384d; border-radius: 4px; min-height: 24px; }
        QSplitter::handle { background: transparent; width: 12px; }
        """
    )


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("MoLoRA Lab")
    _apply_theme(app)
    window = MolorWindow()
    window.show()
    return app.exec()
