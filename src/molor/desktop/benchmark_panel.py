"""Thin native launcher for the repeatable benchmark CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from molor.desktop.panels import ProcessConsole


class BenchmarkPanel(ProcessConsole):
    """Collect benchmark options and delegate execution to ``molor.benchmark.cli``."""

    def __init__(
        self,
        *,
        workspace_root: Path,
        endpoint_source: Callable[[], str] | None = None,
        output_root: Path | None = None,
    ) -> None:
        super().__init__()
        self.workspace_root = workspace_root.expanduser().resolve()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("MODEL BENCHMARKS")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Run the same deterministic suite against the active teacher endpoint and compare "
            "base, router, or forced-expert routes. Benchmark logic and artifacts stay in the CLI."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("card")
        form = QFormLayout(controls)
        form.setContentsMargins(16, 14, 16, 14)
        form.setSpacing(9)

        self.suite = QLineEdit(
            str(self.workspace_root / "benchmarks" / "suites" / "molor-core-v1.json")
        )
        self.suite.setToolTip("Immutable benchmark suite JSON")
        form.addRow("Suite", self._file_row(self.suite, "Choose benchmark suite"))

        self.profile = QComboBox()
        self.profile.setEditable(True)
        self.profile.addItems(["smoke", "quick", "full"])
        self.profile.setCurrentText("quick")
        self.profile.setToolTip("A profile declared by the selected suite")
        form.addRow("Profile", self.profile)

        self.backend = QComboBox()
        self.backend.addItem("Active MoLoRA teacher API", "teacher")
        self.backend.addItem("OpenAI-compatible server", "openai")
        self.backend.currentIndexChanged.connect(self._backend_changed)
        form.addRow("Backend", self.backend)

        default_endpoint = endpoint_source() if endpoint_source is not None else ""
        self.endpoint = QLineEdit(default_endpoint or "http://127.0.0.1:8765")
        self.endpoint.setPlaceholderText("http://127.0.0.1:8765")
        form.addRow("Endpoint", self.endpoint)

        self.model = QLineEdit()
        self.model.setPlaceholderText("Required only for OpenAI-compatible servers")
        self.model.setEnabled(False)
        form.addRow("Server model ID", self.model)

        self.route = QComboBox()
        self.route.setEditable(True)
        self.route.addItems(["auto", "base"])
        self.route.setCurrentText("auto")
        self.route.setToolTip(
            "Use auto, one route name, or a comma-separated weighted route such as expert-a=0.7,expert-b=0.3"
        )
        form.addRow("Route", self.route)

        self.route_top_k = QSpinBox()
        self.route_top_k.setRange(1, 64)
        self.route_top_k.setValue(1)
        self.route_top_k.setToolTip(
            "1 is hard top-1; 2+ applies a real probability-weighted mixture"
        )
        form.addRow("Router top-k", self.route_top_k)

        self.matrix_mode = QCheckBox("Run base + every loaded expert + auto top-1/top-2")
        self.matrix_mode.setToolTip(
            "Builds a route rack scoreboard plus a best-forced-route oracle and router regret"
        )
        self.matrix_mode.toggled.connect(self._matrix_changed)
        form.addRow("Route matrix", self.matrix_mode)

        self.repetitions = QSpinBox()
        self.repetitions.setRange(1, 20)
        self.repetitions.setValue(1)
        form.addRow("Repetitions", self.repetitions)

        self.label = QLineEdit("desktop-benchmark")
        self.label.setPlaceholderText("Name shown in benchmark history")
        form.addRow("Run label", self.label)

        output_root = output_root or self.workspace_root / "runs" / "benchmarks"
        self.output = QLineEdit(str(output_root.resolve()))
        form.addRow("Output root", self._directory_row(self.output, "Choose output folder"))

        self.baseline = QLineEdit()
        self.baseline.setPlaceholderText("Optional prior run directory or summary.json")
        form.addRow("Baseline", self._baseline_row())
        layout.addWidget(controls)

        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open results folder")
        self.open_button.clicked.connect(self.open_results_folder)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.run_button = QPushButton("Start benchmark")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_benchmark)
        buttons.addWidget(self.open_button)
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)

        self.log.setMinimumHeight(230)
        layout.addWidget(self.log, 1)

    def _file_row(self, field: QLineEdit, title: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")

        def choose() -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                title,
                field.text(),
                "JSON files (*.json);;All files (*)",
            )
            if selected:
                field.setText(selected)

        browse.clicked.connect(choose)
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _directory_row(self, field: QLineEdit, title: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")

        def choose() -> None:
            selected = QFileDialog.getExistingDirectory(self, title, field.text())
            if selected:
                field.setText(selected)

        browse.clicked.connect(choose)
        layout.addWidget(field, 1)
        layout.addWidget(browse)
        return row

    def _baseline_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        file_button = QPushButton("File")
        folder_button = QPushButton("Folder")

        def choose_file() -> None:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Choose baseline summary",
                self.baseline.text(),
                "JSON files (*.json);;All files (*)",
            )
            if selected:
                self.baseline.setText(selected)

        def choose_folder() -> None:
            selected = QFileDialog.getExistingDirectory(
                self,
                "Choose baseline run",
                self.baseline.text(),
            )
            if selected:
                self.baseline.setText(selected)

        file_button.clicked.connect(choose_file)
        folder_button.clicked.connect(choose_folder)
        layout.addWidget(self.baseline, 1)
        layout.addWidget(file_button)
        layout.addWidget(folder_button)
        return row

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()

    @Slot(int)
    def _backend_changed(self, _index: int) -> None:
        teacher = self.backend.currentData() == "teacher"
        if not teacher:
            self.matrix_mode.setChecked(False)
        self.matrix_mode.setEnabled(teacher)
        self.route.setEnabled(teacher and not self.matrix_mode.isChecked())
        self.route_top_k.setEnabled(teacher and not self.matrix_mode.isChecked())
        self.model.setEnabled(not teacher)
        if teacher and not self.endpoint.text().strip():
            self.endpoint.setText("http://127.0.0.1:8765")
        elif not teacher and self.endpoint.text().strip() == "http://127.0.0.1:8765":
            self.endpoint.setText("http://127.0.0.1:1234/v1")

    @Slot(bool)
    def _matrix_changed(self, enabled: bool) -> None:
        teacher = self.backend.currentData() == "teacher"
        self.route.setEnabled(teacher and not enabled)
        self.route_top_k.setEnabled(teacher and not enabled)
        self.baseline.setEnabled(not enabled)
        self.run_button.setText("Start route matrix" if enabled else "Start benchmark")

    @Slot()
    def run_benchmark(self) -> None:
        if self.is_running:
            return
        try:
            suite = self._path(self.suite.text().strip())
            if not suite.is_file():
                raise FileNotFoundError(f"Suite does not exist: {suite}")

            profile = self.profile.currentText().strip()
            endpoint = self.endpoint.text().strip()
            route = self.route.currentText().strip()
            backend = str(self.backend.currentData())
            label = self.label.text().strip()
            if not profile:
                raise ValueError("Profile is required")
            if not endpoint:
                raise ValueError("Teacher endpoint is required")
            if backend == "teacher" and not route and not self.matrix_mode.isChecked():
                raise ValueError("Route is required")
            if backend == "openai" and not self.model.text().strip():
                raise ValueError("Server model ID is required for an OpenAI-compatible backend")
            if not label:
                raise ValueError("Run label is required")

            output_value = self.output.text().strip()
            if not output_value:
                raise ValueError("Output root is required")
            output = self._path(output_value)
            output.mkdir(parents=True, exist_ok=True)

            matrix_mode = self.matrix_mode.isChecked()
            arguments = [
                "matrix" if matrix_mode else "run",
                "--suite",
                str(suite),
                "--profile",
                profile,
                "--endpoint",
                endpoint,
                "--label",
                label,
                "--output-root",
                str(output),
                "--repetitions",
                str(self.repetitions.value()),
            ]
            if not matrix_mode:
                arguments.extend(["--backend", backend])
            if backend == "teacher" and not matrix_mode:
                arguments.extend(
                    [
                        "--route",
                        route,
                        "--route-top-k",
                        str(self.route_top_k.value()),
                    ]
                )
            elif backend == "openai":
                arguments.extend(["--model", self.model.text().strip()])
            baseline_value = self.baseline.text().strip()
            if baseline_value and not matrix_mode:
                baseline = self._path(baseline_value)
                if not baseline.exists():
                    raise FileNotFoundError(f"Baseline does not exist: {baseline}")
                arguments.extend(["--baseline", str(baseline)])
        except (OSError, ValueError) as exc:
            self._append_log(f"[cannot start] {type(exc).__name__}: {exc}")
            return

        self.launch("molor.benchmark.cli", arguments)

    @Slot()
    def open_results_folder(self) -> None:
        output_value = self.output.text().strip()
        if not output_value:
            self._append_log("[cannot open] Output root is empty")
            return
        try:
            output = self._path(output_value)
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._append_log(f"[cannot open] {type(exc).__name__}: {exc}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(output))):
            self._append_log(f"[cannot open] Desktop could not open {output}")


__all__ = ["BenchmarkPanel"]
