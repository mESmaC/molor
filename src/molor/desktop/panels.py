"""Native experiment panels that launch the existing local command-line tools."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Signal, Slot
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from molor.feedback import FeedbackStore, GenerationRecord


QWEN_ALL = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class AnnotationPanel(QWidget):
    """Grade generations in-place and export them for the next adapter experiment."""

    def __init__(self, store: FeedbackStore) -> None:
        super().__init__()
        self.store = store
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("ANNOTATION + REWARD")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel("Score what just happened, correct it, or compare two routed outputs.")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Generation"))
        self.generation = QComboBox()
        self.generation.currentIndexChanged.connect(self._selection_changed)
        selector_row.addWidget(self.generation, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        selector_row.addWidget(refresh)
        layout.addLayout(selector_row)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(145)
        layout.addWidget(self.preview)
        self.teacher_summary = QLabel("No model-teacher reviews for this generation.")
        self.teacher_summary.setObjectName("muted")
        self.teacher_summary.setWordWrap(True)
        layout.addWidget(self.teacher_summary)

        scores = QFrame()
        scores.setObjectName("card")
        score_form = QFormLayout(scores)
        score_form.setContentsMargins(16, 14, 16, 14)
        score_row = QWidget()
        score_layout = QHBoxLayout(score_row)
        score_layout.setContentsMargins(0, 0, 0, 0)
        self.reward = _score_spin()
        self.correctness = _score_spin()
        self.style = _score_spin()
        self.reasoning = _score_spin()
        for name, field in (
            ("Overall", self.reward),
            ("Correct", self.correctness),
            ("Style", self.style),
            ("Reason", self.reasoning),
        ):
            score_layout.addWidget(QLabel(name))
            score_layout.addWidget(field)
        score_layout.addStretch()
        score_form.addRow("Scores (-2..2)", score_row)
        self.notes = QPlainTextEdit()
        self.notes.setMaximumHeight(70)
        self.notes.setPlaceholderText("What changed, failed, or surprised you?")
        score_form.addRow("Notes", self.notes)
        self.corrected = QPlainTextEdit()
        self.corrected.setMaximumHeight(100)
        self.corrected.setPlaceholderText("Optional better answer; this becomes usable SFT data.")
        score_form.addRow("Corrected answer", self.corrected)
        save_row = QHBoxLayout()
        self.grade_status = QLabel()
        self.grade_status.setObjectName("muted")
        save_row.addWidget(self.grade_status)
        save_row.addStretch()
        save = QPushButton("Save grade")
        save.setObjectName("primary")
        save.clicked.connect(self.save_grade)
        save_row.addWidget(save)
        score_form.addRow("", save_row)
        layout.addWidget(scores)

        preference = QFrame()
        preference.setObjectName("card")
        preference_form = QFormLayout(preference)
        preference_form.setContentsMargins(16, 14, 16, 14)
        self.chosen = QComboBox()
        self.rejected = QComboBox()
        preference_form.addRow("Chosen", self.chosen)
        preference_form.addRow("Rejected", self.rejected)
        self.preference_notes = QLineEdit()
        self.preference_notes.setPlaceholderText("Optional comparison note")
        preference_form.addRow("Why", self.preference_notes)
        preference_row = QHBoxLayout()
        self.preference_status = QLabel("Pairwise rows must share the same prompt/messages.")
        self.preference_status.setObjectName("muted")
        preference_row.addWidget(self.preference_status)
        preference_row.addStretch()
        save_preference = QPushButton("Save preference")
        save_preference.clicked.connect(self.save_preference)
        preference_row.addWidget(save_preference)
        preference_form.addRow("", preference_row)
        layout.addWidget(preference)

        export_row = QHBoxLayout()
        self.export_dir = QLineEdit(str(Path("runs/feedback/exports").resolve()))
        export_row.addWidget(self.export_dir, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_export)
        export_row.addWidget(browse)
        for title, kind in (
            ("Human Rewards", "rewards"),
            ("Teacher RL", "teacher"),
            ("Training SFT", "sft"),
            ("Preferences", "prefs"),
        ):
            button = QPushButton(f"Export {title}")
            button.clicked.connect(
                lambda _checked=False, export_kind=kind: self.export(export_kind)
            )
            export_row.addWidget(button)
        layout.addLayout(export_row)
        layout.addStretch()
        self.refresh()

    @property
    def selected_generation_id(self) -> str | None:
        value = self.generation.currentData()
        return str(value) if value else None

    def generation_recorded(self, generation_id: str) -> None:
        self.refresh(select_id=generation_id)

    def quick_grade(self, generation_id: str, reward: int) -> GenerationRecord:
        existing = self.store.get_generation(generation_id)
        if existing is None:
            raise KeyError(f"Unknown generation {generation_id}")
        record = self.store.grade_generation(
            generation_id,
            reward=reward,
            dimensions=existing.dimensions,
            notes=existing.notes,
            corrected_answer=existing.corrected_answer,
        )
        self.refresh(select_id=generation_id)
        self.grade_status.setText(f"Saved {reward:+d} · {generation_id[:10]}")
        return record

    @Slot()
    def refresh(self, *, select_id: str | None = None) -> None:
        selected = select_id or self.selected_generation_id
        records = self.store.list_generations(limit=100)
        for combo in (self.generation, self.chosen, self.rejected):
            combo.blockSignals(True)
            combo.clear()
            for record in records:
                combo.addItem(_record_label(record), record.generation_id)
            combo.blockSignals(False)
        if selected:
            index = self.generation.findData(selected)
            if index >= 0:
                self.generation.setCurrentIndex(index)
        if self.rejected.count() > 1:
            self.rejected.setCurrentIndex(1)
        self._selection_changed()

    @Slot()
    def _selection_changed(self) -> None:
        generation_id = self.selected_generation_id
        record = self.store.get_generation(generation_id) if generation_id else None
        if record is None:
            self.preview.clear()
            self.teacher_summary.setText("No model-teacher reviews for this generation.")
            return
        self.preview.setPlainText(f"PROMPT\n{record.prompt}\n\nRESPONSE\n{record.response}")
        self.reward.setValue(int(record.reward or 0))
        self.correctness.setValue(int(record.dimensions.get("correctness", 0)))
        self.style.setValue(int(record.dimensions.get("style", 0)))
        self.reasoning.setValue(int(record.dimensions.get("reasoning", 0)))
        self.notes.setPlainText(record.notes)
        self.corrected.setPlainText(record.corrected_answer or "")
        consensus = self.store.teacher_consensus(record.generation_id)
        count = int(consensus["annotation_count"])
        if count:
            overall = float(consensus["weighted_overall"])
            disagreement = float(consensus["disagreement"])
            dimensions = ", ".join(
                f"{name} {value:+.2f}"
                for name, value in dict(consensus["dimensions"]).items()
            )
            self.teacher_summary.setText(
                f"Model teachers: {count} · consensus {overall:+.2f} · "
                f"disagreement {disagreement:.2f}"
                + (f" · {dimensions}" if dimensions else "")
            )
        else:
            self.teacher_summary.setText("No model-teacher reviews for this generation.")

    @Slot()
    def save_grade(self) -> None:
        generation_id = self.selected_generation_id
        if generation_id is None:
            self.grade_status.setText("Generate something first")
            return
        try:
            self.store.grade_generation(
                generation_id,
                reward=self.reward.value(),
                dimensions={
                    "correctness": self.correctness.value(),
                    "style": self.style.value(),
                    "reasoning": self.reasoning.value(),
                },
                notes=self.notes.toPlainText().strip(),
                corrected_answer=self.corrected.toPlainText().strip() or None,
            )
        except Exception as exc:
            self.grade_status.setText(f"{type(exc).__name__}: {exc}")
            return
        self.grade_status.setText(f"Saved · {generation_id[:10]}")
        self.refresh(select_id=generation_id)

    @Slot()
    def save_preference(self) -> None:
        chosen = self.chosen.currentData()
        rejected = self.rejected.currentData()
        if not chosen or not rejected:
            self.preference_status.setText("Need two generations")
            return
        try:
            preference_id = self.store.record_preference(
                str(chosen),
                str(rejected),
                notes=self.preference_notes.text().strip(),
            )
        except Exception as exc:
            self.preference_status.setText(f"{type(exc).__name__}: {exc}")
            return
        self.preference_status.setText(f"Saved · {preference_id[:10]}")

    @Slot()
    def _browse_export(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Export feedback", self.export_dir.text()
        )
        if directory:
            self.export_dir.setText(directory)

    def export(self, kind: str) -> None:
        directory = Path(self.export_dir.text().strip()).expanduser()
        try:
            if kind == "rewards":
                result = self.store.export_reward_jsonl(directory / "rewards.jsonl")
            elif kind == "teacher":
                result = self.store.export_teacher_reward_jsonl(
                    directory / "teacher_rewards.jsonl"
                )
            elif kind == "sft":
                result = self.store.export_training_sft_jsonl(directory / "training_sft.jsonl")
            else:
                result = self.store.export_preference_jsonl(directory / "preferences.jsonl")
        except Exception as exc:
            self.grade_status.setText(f"Export failed: {type(exc).__name__}: {exc}")
            return
        self.grade_status.setText(f"Exported {result.rows} rows · {result.path}")


class ProcessConsole(QWidget):
    """Shared QProcess plumbing for the two long-running experiment panels."""

    process_started = Signal()
    process_finished = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.started.connect(self._started)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(10_000)
        self.log.setPlaceholderText("Process output appears here.")
        self.run_button: QPushButton | None = None
        self.stop_button: QPushButton | None = None

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.ProcessState.NotRunning

    def launch(self, module: str, arguments: list[str]) -> None:
        if self.is_running:
            return
        self.log.clear()
        self._append_log(f"> {sys.executable} -m {module} {' '.join(arguments)}")
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-u", "-m", module, *arguments])
        self.process.setWorkingDirectory(str(Path.cwd()))
        self.process.start()

    @Slot()
    def stop(self) -> None:
        if not self.is_running:
            return
        self._append_log("\n[stopping]")
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    @Slot()
    def _kill_if_running(self) -> None:
        if self.is_running:
            self.process.kill()

    @Slot()
    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            cursor = self.log.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(data)
            self.log.setTextCursor(cursor)
            self.log.ensureCursorVisible()

    @Slot()
    def _started(self) -> None:
        if self.run_button is not None:
            self.run_button.setEnabled(False)
        if self.stop_button is not None:
            self.stop_button.setEnabled(True)
        self.process_started.emit()

    @Slot(int, QProcess.ExitStatus)
    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        self._append_log(f"\n[finished with exit code {exit_code}]")
        if self.run_button is not None:
            self.run_button.setEnabled(True)
        if self.stop_button is not None:
            self.stop_button.setEnabled(False)
        self.process_finished.emit(exit_code)

    @Slot(QProcess.ProcessError)
    def _process_error(self, _error: QProcess.ProcessError) -> None:
        self._append_log(f"\n[process error] {self.process.errorString()}")

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)


class TeacherPanel(ProcessConsole):
    """Launch and explain the loopback API used by external model teachers."""

    def __init__(
        self,
        *,
        model_source: Callable[[], str],
        expert_paths_source: Callable[[], tuple[str, ...]],
        router_source: Callable[[], str],
        before_gpu_job: Callable[[str], bool],
    ) -> None:
        super().__init__()
        self.model_source = model_source
        self.expert_paths_source = expert_paths_source
        self.router_source = router_source
        self.before_gpu_job = before_gpu_job

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("TEACHER / STUDENT RL INTERFACE")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "A loopback-only tool API for Codex, Claude, GLM, Kimi, or any other agent. "
            "It records independent grades instead of replacing your human annotations."
        )
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("card")
        form = QFormLayout(controls)
        form.setContentsMargins(16, 14, 16, 14)
        self.port = QSpinBox()
        self.port.setRange(1024, 65535)
        self.port.setValue(8765)
        form.addRow("Loopback port", self.port)
        self.database = QLineEdit(str(Path("runs/feedback/feedback.sqlite3").resolve()))
        form.addRow("Feedback database", _browse_row(self.database, self, file_mode=True))
        self.eager_load = QCheckBox("Load the student as soon as the server starts")
        self.eager_load.setChecked(False)
        form.addRow("Model loading", self.eager_load)
        model_note = QLabel(
            "Uses the model, every checked resident expert, and router currently selected "
            "on the Chat tab."
        )
        model_note.setObjectName("muted")
        model_note.setWordWrap(True)
        form.addRow("Student", model_note)
        layout.addWidget(controls)

        instructions = QFrame()
        instructions.setObjectName("card")
        instructions_layout = QVBoxLayout(instructions)
        instructions_layout.setContentsMargins(16, 14, 16, 14)
        instructions_layout.addWidget(QLabel("MODEL TOOL CONTRACT"))
        self.contract = QPlainTextEdit()
        self.contract.setReadOnly(True)
        self.contract.setMaximumHeight(175)
        self.contract.setPlainText(self._contract_text())
        instructions_layout.addWidget(self.contract)
        copy_button = QPushButton("Copy agent instructions")
        copy_button.clicked.connect(self._copy_contract)
        instructions_layout.addWidget(copy_button)
        layout.addWidget(instructions)

        buttons = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        self.run_button = QPushButton("Start teacher API")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_server)
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)
        self.log.setMinimumHeight(180)
        layout.addWidget(self.log, 1)

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port.value()}"

    def _contract_text(self) -> str:
        endpoint = self.endpoint()
        return (
            f"Endpoint: {endpoint}\n"
            f"Health:   {endpoint}/health\n"
            f"Schemas:  {endpoint}/v1/capabilities\n"
            f"OpenAPI:  {endpoint}/openapi.json\n\n"
            "CLI tool:\n"
            ".\\.venv\\Scripts\\molor-teacher.exe schema\n"
            "$request = @{ prompt = 'Explain volatile in embedded C'; candidates = 2 } "
            "| ConvertTo-Json\n"
            "$request | .\\.venv\\Scripts\\molor-teacher.exe invoke ask_student"
        )

    @Slot()
    def _copy_contract(self) -> None:
        self.contract.setPlainText(self._contract_text())
        QApplication.clipboard().setText(
            "You can evaluate my local student through the MoLoRA teacher API.\n"
            f"Base URL: {self.endpoint()}\n"
            "First read GET /v1/capabilities for the rubric and exact JSON tool schemas. "
            "Use ask_student to generate candidates, grade_student for independent scores and "
            "corrections, and prefer_student_output for pairwise ranking."
        )
        self._append_log("Copied teacher-agent instructions to the clipboard.")

    @Slot()
    def run_server(self) -> None:
        if self.is_running or not self.before_gpu_job("the teacher API"):
            return
        model = self.model_source().strip()
        if not model:
            self._append_log("[cannot start] Enter a model on the Chat tab")
            return
        arguments = [
            "serve",
            "--model",
            model,
            "--db",
            self.database.text().strip(),
            "--port",
            str(self.port.value()),
        ]
        expert_paths = tuple(path.strip() for path in self.expert_paths_source() if path.strip())
        if expert_paths:
            arguments.extend(["--expert-a", expert_paths[0]])
        if len(expert_paths) > 1:
            arguments.extend(["--expert-b", expert_paths[1]])
        for expert_path in expert_paths[2:]:
            arguments.extend(["--expert", expert_path])
        router = self.router_source().strip()
        if router:
            arguments.extend(["--router", router])
        if self.eager_load.isChecked():
            arguments.append("--eager-load")
        self.contract.setPlainText(self._contract_text())
        self.launch("molor.teacher.cli", arguments)


class TrainingPanel(ProcessConsole):
    """Quick-recipe builder plus an advanced path to the existing trainer CLI."""

    def __init__(
        self,
        *,
        model_source: Callable[[], str],
        before_gpu_job: Callable[[str], bool],
        export_positive_annotations: Callable[[Path], int],
        export_preference_annotations: Callable[[Path], int],
    ) -> None:
        super().__init__()
        self.model_source = model_source
        self.before_gpu_job = before_gpu_job
        self.export_positive_annotations = export_positive_annotations
        self.export_preference_annotations = export_preference_annotations

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("EXPERT TRAINING")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "Build one small adapter quickly, or point the existing trainer at a full recipe."
        )
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        modes = QTabWidget()
        modes.setObjectName("innerTabs")
        modes.addTab(self._build_quick(), "Quick Train")
        modes.addTab(self._build_advanced(), "Existing Recipe")
        self.mode_tabs = modes
        layout.addWidget(modes)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Train expert")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)
        self.log.setMinimumHeight(210)
        layout.addWidget(self.log, 1)

    def _build_quick(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(10, 12, 10, 10)
        form.setSpacing(9)

        self.quick_name = QLineEdit("new_expert")
        form.addRow("Expert name", self.quick_name)
        self.quick_dataset = QLineEdit()
        self.quick_dataset.setPlaceholderText("JSONL chat dataset")
        form.addRow("Dataset", _browse_row(self.quick_dataset, self, file_mode=True))
        annotations = QPushButton("Use positive annotations")
        annotations.clicked.connect(self._use_annotations)
        form.addRow("", annotations)
        self.quick_output = QLineEdit(str((Path("checkpoints") / "quick").resolve()))
        form.addRow("Output root", _browse_row(self.quick_output, self, file_mode=False))

        compact = QWidget()
        compact_row = QHBoxLayout(compact)
        compact_row.setContentsMargins(0, 0, 0, 0)
        self.quick_rank = QSpinBox()
        self.quick_rank.setRange(1, 256)
        self.quick_rank.setValue(8)
        self.quick_alpha = QDoubleSpinBox()
        self.quick_alpha.setRange(0.1, 1024)
        self.quick_alpha.setValue(16)
        compact_row.addWidget(QLabel("Rank"))
        compact_row.addWidget(self.quick_rank)
        compact_row.addWidget(QLabel("Alpha"))
        compact_row.addWidget(self.quick_alpha)
        compact_row.addStretch()
        form.addRow("LoRA", compact)

        compact = QWidget()
        compact_row = QHBoxLayout(compact)
        compact_row.setContentsMargins(0, 0, 0, 0)
        self.quick_epochs = QSpinBox()
        self.quick_epochs.setRange(1, 100)
        self.quick_epochs.setValue(4)
        self.quick_lr = QDoubleSpinBox()
        self.quick_lr.setDecimals(7)
        self.quick_lr.setRange(0.0000001, 0.1)
        self.quick_lr.setValue(0.0001)
        compact_row.addWidget(QLabel("Epochs"))
        compact_row.addWidget(self.quick_epochs)
        compact_row.addWidget(QLabel("LR"))
        compact_row.addWidget(self.quick_lr)
        compact_row.addStretch()
        form.addRow("Optimizer", compact)

        self.quick_max_length = QSpinBox()
        self.quick_max_length.setRange(64, 32_768)
        self.quick_max_length.setSingleStep(64)
        self.quick_max_length.setValue(384)
        form.addRow("Max length", self.quick_max_length)

        compact = QWidget()
        compact_row = QHBoxLayout(compact)
        compact_row.setContentsMargins(0, 0, 0, 0)
        self.quick_micro_batch = QSpinBox()
        self.quick_micro_batch.setRange(1, 64)
        self.quick_micro_batch.setValue(1)
        self.quick_grad_accum = QSpinBox()
        self.quick_grad_accum.setRange(1, 256)
        self.quick_grad_accum.setValue(4)
        compact_row.addWidget(QLabel("Microbatch"))
        compact_row.addWidget(self.quick_micro_batch)
        compact_row.addWidget(QLabel("Accumulate"))
        compact_row.addWidget(self.quick_grad_accum)
        compact_row.addStretch()
        form.addRow("Batching", compact)

        self.quick_targets = QComboBox()
        self.quick_targets.addItem("All Qwen projections", QWEN_ALL)
        self.quick_targets.addItem("Attention only", ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.quick_targets.addItem("Q + V only", ("q_proj", "v_proj"))
        self.quick_targets.addItem("MLP only", ("gate_proj", "up_proj", "down_proj"))
        form.addRow("Target preset", self.quick_targets)
        return panel

    def _build_advanced(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        form.setContentsMargins(10, 12, 10, 10)
        self.recipe_path = QLineEdit(
            str(
                next(
                    (
                        path.resolve()
                        for path in (
                            Path(
                                "experiments/first-real-experts/finance-verified-recipe.json"
                            ),
                            Path("experiments/first-real-experts/finance-recipe.json"),
                            Path("experiments/qwen3-1.7b-two-styles/recipe.json"),
                        )
                        if path.is_file()
                    ),
                    Path(
                        "experiments/first-real-experts/finance-verified-recipe.json"
                    ).resolve(),
                )
            )
        )
        form.addRow("Recipe", _browse_row(self.recipe_path, self, file_mode=True))
        self.expert_selector = QLineEdit()
        self.expert_selector.setPlaceholderText("Blank trains every expert in the recipe")
        form.addRow("Only expert", self.expert_selector)
        self.preference_recipe = QLineEdit(
            str(Path("experiments/qwen3-1.7b-two-styles/preference_recipe.json").resolve())
        )
        form.addRow(
            "Preference recipe",
            _browse_row(self.preference_recipe, self, file_mode=True),
        )
        self.preference_button = QPushButton("Train preference / DPO expert")
        self.preference_button.clicked.connect(self.run_preference_training)
        form.addRow("", self.preference_button)
        return panel

    @Slot()
    def _use_annotations(self) -> None:
        target = Path("runs/annotation-datasets") / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-positive.jsonl"
        )
        try:
            count = self.export_positive_annotations(target.resolve())
        except Exception as exc:
            self._append_log(f"[annotation export failed] {type(exc).__name__}: {exc}")
            return
        if count == 0:
            self.quick_dataset.clear()
            self._append_log(
                "No positive or corrected annotations yet. Grade a response +1/+2 or add a "
                "corrected answer first."
            )
            return
        self.quick_dataset.setText(str(target.resolve()))
        self._append_log(f"Exported {count} positive/corrected examples to {target}")

    @Slot()
    def run_training(self) -> None:
        if self.is_running or not self.before_gpu_job("expert training"):
            return
        try:
            if self.mode_tabs.currentIndex() == 0:
                recipe = self._write_quick_recipe()
                arguments = [str(recipe), "--local-files-only"]
            else:
                recipe = Path(self.recipe_path.text().strip()).expanduser().resolve()
                if not recipe.is_file():
                    raise FileNotFoundError(f"Recipe does not exist: {recipe}")
                arguments = [str(recipe), "--local-files-only"]
                expert = self.expert_selector.text().strip()
                if expert:
                    arguments.extend(["--expert", expert])
        except Exception as exc:
            self._append_log(f"[cannot start] {type(exc).__name__}: {exc}")
            return
        self.launch("molor.training.cli", arguments)

    @Slot()
    def run_preference_training(self) -> None:
        if self.is_running:
            return
        recipe = Path(self.preference_recipe.text().strip()).expanduser().resolve()
        if not recipe.is_file():
            self._append_log(f"[cannot start] Preference recipe does not exist: {recipe}")
            return
        try:
            payload = json.loads(recipe.read_text(encoding="utf-8"))
            dataset_value = payload.get("dataset")
            if not isinstance(dataset_value, str) or not dataset_value:
                raise ValueError("preference recipe has no dataset path")
            dataset = (recipe.parent / dataset_value).resolve()
            rows = self.export_preference_annotations(dataset)
        except Exception as exc:
            self._append_log(f"[preference export failed] {type(exc).__name__}: {exc}")
            return
        if rows == 0:
            self._append_log(
                "No pairwise preferences yet. Regenerate a prompt, choose the better response "
                "in Annotation / Reward, and save the preference first."
            )
            return
        self._append_log(f"Exported {rows} fresh preference pairs to {dataset}")
        if not self.before_gpu_job("preference training"):
            return
        self.launch(
            "molor.training.preference_cli",
            ["--local-files-only", str(recipe)],
        )

    @Slot()
    def _started(self) -> None:
        super()._started()
        self.preference_button.setEnabled(False)

    @Slot(int, QProcess.ExitStatus)
    def _finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        super()._finished(exit_code, status)
        self.preference_button.setEnabled(True)

    def _write_quick_recipe(self) -> Path:
        name = self.quick_name.text().strip()
        if not name or any(not character.isalnum() and character not in "_-" for character in name):
            raise ValueError("Expert name may contain letters, numbers, underscores, and hyphens")
        dataset = Path(self.quick_dataset.text().strip()).expanduser().resolve()
        if not dataset.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset}")
        output = Path(self.quick_output.text().strip()).expanduser().resolve()
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
        recipe_dir = Path("runs/recipes").resolve()
        recipe_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = recipe_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{safe_name}.json")
        payload = {
            "model_id": self.model_source(),
            "output_dir": str(output),
            "dtype": "bfloat16",
            "device": "cuda",
            "target_modules": list(self.quick_targets.currentData()),
            "rank": self.quick_rank.value(),
            "alpha": self.quick_alpha.value(),
            "dropout": 0.05,
            "max_length": self.quick_max_length.value(),
            "epochs": self.quick_epochs.value(),
            "learning_rate": self.quick_lr.value(),
            "micro_batch_size": self.quick_micro_batch.value(),
            "gradient_accumulation_steps": self.quick_grad_accum.value(),
            "gradient_checkpointing": True,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "experts": [{"name": name, "dataset": str(dataset)}],
        }
        recipe_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._append_log(f"Wrote quick recipe: {recipe_path}")
        return recipe_path


class ContextLabPanel(ProcessConsole):
    """Runtime context controls plus a launcher for cache-mode comparisons."""

    def __init__(
        self,
        *,
        model_source: Callable[[], str],
        before_gpu_job: Callable[[str], bool],
    ) -> None:
        super().__init__()
        self.model_source = model_source
        self.before_gpu_job = before_gpu_job
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        title = QLabel("CONTEXT + KV CACHE LAB")
        title.setObjectName("panelTitle")
        layout.addWidget(title)
        subtitle = QLabel(
            "These settings affect chat immediately. The lab runs the same prompt across cache modes."
        )
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        controls = QFrame()
        controls.setObjectName("card")
        form = QFormLayout(controls)
        form.setContentsMargins(16, 14, 16, 14)
        self.budget_preset = QComboBox()
        for label, value in (
            ("Auto / model limit", None),
            ("4K", 4096),
            ("8K", 8192),
            ("16K", 16384),
            ("32K", 32768),
            ("Custom", "custom"),
        ):
            self.budget_preset.addItem(label, value)
        self.budget_preset.currentIndexChanged.connect(self._budget_changed)
        form.addRow("Chat context budget", self.budget_preset)
        self.custom_budget = QSpinBox()
        self.custom_budget.setRange(256, 1_048_576)
        self.custom_budget.setSingleStep(1024)
        self.custom_budget.setValue(40960)
        self.custom_budget.setEnabled(False)
        form.addRow("Custom tokens", self.custom_budget)
        self.kv_mode = QComboBox()
        self.kv_mode.addItem("Dynamic GPU cache", "dynamic")
        self.kv_mode.addItem("Hybrid CPU ↔ GPU exact cache", "offloaded")
        self.kv_mode.addItem("Cache disabled", "disabled")
        form.addRow("Chat KV mode", self.kv_mode)
        self.snapshot_dir = QLineEdit(str(Path("runs/kv-snapshots").resolve()))
        form.addRow("NVMe snapshot dir", _browse_row(self.snapshot_dir, self, file_mode=False))
        layout.addWidget(controls)

        lab = QFrame()
        lab.setObjectName("card")
        lab_form = QFormLayout(lab)
        lab_form.setContentsMargins(16, 14, 16, 14)
        self.lab_prompt = QPlainTextEdit(
            "Explain how a mixture of LoRA adapters can change a frozen model's behavior."
        )
        self.lab_prompt.setMaximumHeight(90)
        lab_form.addRow("Lab prompt", self.lab_prompt)
        self.prefill_tokens = QSpinBox()
        self.prefill_tokens.setRange(1, 1_000_000)
        self.prefill_tokens.setSingleStep(1024)
        self.prefill_tokens.setValue(8192)
        lab_form.addRow("Synthetic prefill", self.prefill_tokens)
        self.lab_max_tokens = QSpinBox()
        self.lab_max_tokens.setRange(1, 8192)
        self.lab_max_tokens.setValue(128)
        lab_form.addRow("New tokens", self.lab_max_tokens)
        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.mode_checks: dict[str, QCheckBox] = {}
        for mode in ("dynamic", "offloaded", "disabled"):
            check = QCheckBox(mode)
            check.setChecked(True)
            self.mode_checks[mode] = check
            mode_layout.addWidget(check)
        mode_layout.addStretch()
        lab_form.addRow("Compare modes", mode_row)
        layout.addWidget(lab)

        buttons = QHBoxLayout()
        self.run_button = QPushButton("Run cache lab")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_lab)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        buttons.addStretch()
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.run_button)
        layout.addLayout(buttons)
        self.log.setMinimumHeight(190)
        layout.addWidget(self.log, 1)

    def context_token_budget(self) -> int | None:
        value = self.budget_preset.currentData()
        if value == "custom":
            return self.custom_budget.value()
        return value

    def cache_mode(self) -> str:
        return str(self.kv_mode.currentData())

    @Slot()
    def _budget_changed(self) -> None:
        self.custom_budget.setEnabled(self.budget_preset.currentData() == "custom")

    @Slot()
    def run_lab(self) -> None:
        if self.is_running or not self.before_gpu_job("the cache lab"):
            return
        modes = [name for name, check in self.mode_checks.items() if check.isChecked()]
        if not modes:
            self._append_log("[cannot start] Select at least one cache mode")
            return
        budget = self.context_token_budget() or 40960
        arguments = [
            "--model",
            self.model_source(),
            "--prompt",
            self.lab_prompt.toPlainText().strip(),
            "--max-new-tokens",
            str(self.lab_max_tokens.value()),
            "--context-token-budget",
            str(budget),
            "--modes",
            *modes,
            "--snapshot-directory",
            self.snapshot_dir.text().strip(),
            "--prefill-tokens",
            str(self.prefill_tokens.value()),
            "--benchmark-prompt-tokens",
            str(self.prefill_tokens.value()),
        ]
        self.launch("molor.experimental.cache_lab", arguments)


def _browse_row(field: QLineEdit, parent: QWidget, *, file_mode: bool) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    browse = QPushButton("Browse")

    def choose() -> None:
        if file_mode:
            selected, _ = QFileDialog.getOpenFileName(parent, "Choose file", field.text())
        else:
            selected = QFileDialog.getExistingDirectory(parent, "Choose directory", field.text())
        if selected:
            field.setText(selected)

    browse.clicked.connect(choose)
    layout.addWidget(field, 1)
    layout.addWidget(browse)
    return container


def _score_spin() -> QSpinBox:
    field = QSpinBox()
    field.setRange(-2, 2)
    field.setValue(0)
    return field


def _record_label(record: GenerationRecord) -> str:
    reward = "ungraded" if record.reward is None else f"{record.reward:+g}"
    prompt = " ".join(record.prompt.split())
    return f"{record.created_at[5:16]} · {reward} · {prompt[:72]}"


__all__ = ["AnnotationPanel", "ContextLabPanel", "TrainingPanel"]
