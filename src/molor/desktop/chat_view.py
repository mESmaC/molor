"""Native rich chat transcript widgets for the desktop application.

Consumer contract
-----------------
``ChatTranscript.append_plain`` adds a user or runtime message as escaped,
selectable plain text. ``ChatTranscript.begin_assistant`` adds an assistant
message and returns its ``AssistantMessageWidget``.  A streaming consumer keeps
the canonical/raw model response elsewhere, parses any leading reasoning trace,
and passes the *complete accumulated* answer/reasoning strings to the returned
widget's setters.  Markdown must not be supplied one token fragment at a time,
because constructs such as fenced code blocks can span fragments.

Reasoning is collapsed by default.  Markdown is parsed by Qt with GitHub
extensions and embedded HTML disabled.  Document resources are never loaded;
only explicitly clicked HTTP(S) links are handed to the operating system.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QByteArray, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QTextDocument
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


_MARKDOWN_FEATURES = (
    QTextDocument.MarkdownFeature.MarkdownDialectGitHub
    | QTextDocument.MarkdownFeature.MarkdownNoHTML
)


class SafeTextPane(QTextBrowser):
    """Auto-height text pane that cannot fetch Markdown resources."""

    heightChanged = Signal()

    def __init__(self, *, object_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setReadOnly(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.document().setDocumentMargin(0)
        self.document().setDefaultStyleSheet(
            "a { color: #8aa7ff; } "
            "code { color: #d7e1ff; background-color: #171d29; } "
            "pre { color: #d7e1ff; background-color: #171d29; } "
            "blockquote { color: #aab4c8; }"
        )
        self.anchorClicked.connect(self._open_safe_link)
        self.document().documentLayout().documentSizeChanged.connect(
            lambda _size: self._sync_height()
        )
        self._sync_height()

    def set_markdown(self, content: str) -> None:
        self.document().setMarkdown(content, _MARKDOWN_FEATURES)
        self._sync_height()

    def set_plain(self, content: str) -> None:
        self.setPlainText(content)
        self._sync_height()

    def loadResource(self, resource_type: int, name: QUrl) -> QByteArray:  # noqa: N802
        """Block automatic image, stylesheet, and local-file resource reads."""

        del resource_type, name
        return QByteArray()

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_height)

    def _sync_height(self) -> None:
        width = max(1, self.viewport().width())
        if not math.isclose(self.document().textWidth(), width, abs_tol=1.0):
            self.document().setTextWidth(width)
        document_height = math.ceil(self.document().documentLayout().documentSize().height())
        desired = max(1, document_height + 2 * self.frameWidth())
        if desired != self.height():
            self.setFixedHeight(desired)
            self.heightChanged.emit()

    @staticmethod
    def _open_safe_link(url: QUrl) -> None:
        if (
            url.isValid()
            and url.scheme().lower() in {"http", "https"}
            and bool(url.host())
        ):
            QDesktopServices.openUrl(url)


class PlainMessageWidget(QFrame):
    """A role-labelled, escaped plain-text transcript message."""

    heightChanged = Signal()

    def __init__(
        self,
        role: str,
        content: str,
        *,
        kind: str = "user",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(
            {
                "user": "chatMessageUser",
                "runtime": "chatMessageRuntime",
            }.get(kind.lower(), "chatMessagePlain")
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(5)

        label = QLabel(role)
        label.setObjectName("chatRole")
        layout.addWidget(label)

        self.body = SafeTextPane(object_name="chatPlainText")
        self.body.set_plain(content)
        self.body.heightChanged.connect(self.heightChanged)
        layout.addWidget(self.body)


class AssistantMessageWidget(QFrame):
    """Assistant answer with a separately controlled reasoning disclosure."""

    contentAboutToChange = Signal()
    contentChanged = Signal()

    def __init__(self, role: str = "MOLOR", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatMessageAssistant")
        self._reasoning_incomplete = False
        self._reasoning_streaming = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 10)
        layout.setSpacing(6)

        label = QLabel(role)
        label.setObjectName("chatRoleAssistant")
        layout.addWidget(label)

        self.reasoning_toggle = QToolButton()
        self.reasoning_toggle.setObjectName("reasoningToggle")
        self.reasoning_toggle.setCheckable(True)
        self.reasoning_toggle.setChecked(False)
        self.reasoning_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.reasoning_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.reasoning_toggle.setVisible(False)
        self.reasoning_toggle.toggled.connect(self._toggle_reasoning)
        layout.addWidget(self.reasoning_toggle)

        self.reasoning = SafeTextPane(object_name="reasoningPane")
        self.reasoning.setVisible(False)
        self.reasoning.heightChanged.connect(self._notify_content_changed)
        layout.addWidget(self.reasoning)

        self.answer = SafeTextPane(object_name="assistantAnswer")
        self.answer.heightChanged.connect(self._notify_content_changed)
        layout.addWidget(self.answer)
        self._update_reasoning_label()

    def set_answer_markdown(self, content: str) -> None:
        """Replace the answer using the full accumulated Markdown string."""

        self.contentAboutToChange.emit()
        self.answer.set_markdown(content)
        self._notify_content_changed()

    def update_answer_markdown(self, content: str) -> None:
        """Alias emphasizing that ``content`` replaces, rather than appends."""

        self.set_answer_markdown(content)

    def set_reasoning_markdown(
        self,
        content: str,
        *,
        incomplete: bool = False,
        streaming: bool = False,
    ) -> None:
        self._set_reasoning(
            content,
            markdown=True,
            incomplete=incomplete,
            streaming=streaming,
        )

    def set_reasoning_plain(
        self,
        content: str,
        *,
        incomplete: bool = False,
        streaming: bool = False,
    ) -> None:
        self._set_reasoning(
            content,
            markdown=False,
            incomplete=incomplete,
            streaming=streaming,
        )

    def _set_reasoning(
        self,
        content: str,
        *,
        markdown: bool,
        incomplete: bool,
        streaming: bool,
    ) -> None:
        self.contentAboutToChange.emit()
        self._reasoning_incomplete = incomplete
        self._reasoning_streaming = streaming
        if markdown:
            self.reasoning.set_markdown(content)
        else:
            self.reasoning.set_plain(content)
        has_trace = bool(content.strip()) or incomplete or streaming
        self.reasoning_toggle.setVisible(has_trace)
        if not has_trace:
            self.reasoning_toggle.setChecked(False)
            self.reasoning.setVisible(False)
        self._update_reasoning_label()
        self._notify_content_changed()

    def _toggle_reasoning(self, expanded: bool) -> None:
        self.contentAboutToChange.emit()
        self.reasoning_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self.reasoning.setVisible(expanded)
        self._notify_content_changed()

    def _update_reasoning_label(self) -> None:
        suffix = ""
        if self._reasoning_streaming:
            suffix = "  ·  streaming"
        elif self._reasoning_incomplete:
            suffix = "  ·  interrupted"
        self.reasoning_toggle.setText(f"Reasoning{suffix}")

    def _notify_content_changed(self) -> None:
        QTimer.singleShot(0, self.contentChanged.emit)


class ChatTranscript(QScrollArea):
    """Scrollable transcript that follows output only while already at bottom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatTranscript")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._contents = QWidget()
        self._contents.setObjectName("chatTranscriptContents")
        self._layout = QVBoxLayout(self._contents)
        self._layout.setContentsMargins(2, 2, 2, 2)
        self._layout.setSpacing(10)
        self._layout.addStretch(1)
        self.setWidget(self._contents)
        self._follow_after_change = False

        self.setStyleSheet(
            """
            QScrollArea#chatTranscript, QWidget#chatTranscriptContents {
                background: #0c1119;
            }
            QFrame#chatMessageUser, QFrame#chatMessagePlain {
                background: #121925;
                border: 1px solid #222d40;
                border-radius: 7px;
            }
            QFrame#chatMessageRuntime {
                background: #1b1319;
                border: 1px solid #50303b;
                border-radius: 7px;
            }
            QFrame#chatMessageAssistant {
                background: #101722;
                border: 1px solid #26334a;
                border-radius: 7px;
            }
            QLabel#chatRole, QLabel#chatRoleAssistant {
                color: #8a93a8;
                font-size: 8pt;
                font-weight: 700;
            }
            QLabel#chatRoleAssistant { color: #8aa7ff; }
            QTextBrowser#chatPlainText, QTextBrowser#assistantAnswer,
            QTextBrowser#reasoningPane {
                background: transparent;
                border: 0;
                color: #edf1f8;
                padding: 0;
                selection-background-color: #4d68c9;
            }
            QTextBrowser#reasoningPane {
                color: #adb7ca;
                background: #0c121c;
                border-left: 2px solid #34435f;
                padding-left: 9px;
            }
            QToolButton#reasoningToggle {
                color: #aab5ca;
                background: transparent;
                border: 0;
                padding: 3px 0;
                text-align: left;
            }
            QToolButton#reasoningToggle:hover { color: #d4dcf0; }
            """
        )

    def append_plain(
        self,
        role: str,
        content: str,
        kind: str = "user",
    ) -> PlainMessageWidget:
        follow = self._is_at_bottom()
        message = PlainMessageWidget(role, content, kind=kind)
        self._layout.insertWidget(self._layout.count() - 1, message)
        message.heightChanged.connect(lambda: self._child_changed(follow))
        self._scroll_after_layout(follow)
        return message

    def begin_assistant(self, role: str = "MOLOR") -> AssistantMessageWidget:
        follow = self._is_at_bottom()
        message = AssistantMessageWidget(role)
        self._layout.insertWidget(self._layout.count() - 1, message)
        message.contentAboutToChange.connect(self._prepare_child_change)
        message.contentChanged.connect(self._finish_child_change)
        self._scroll_after_layout(follow)
        return message

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        follow = self._is_at_bottom()
        super().resizeEvent(event)
        self._scroll_after_layout(follow)

    def _is_at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.maximum() - bar.value() <= 2

    def _prepare_child_change(self) -> None:
        self._follow_after_change = self._is_at_bottom()

    def _finish_child_change(self) -> None:
        follow = self._follow_after_change
        self._follow_after_change = False
        self._scroll_after_layout(follow)

    def _child_changed(self, follow: bool) -> None:
        self._scroll_after_layout(follow)

    def _scroll_after_layout(self, follow: bool) -> None:
        if not follow:
            return

        def scroll() -> None:
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

        QTimer.singleShot(0, scroll)


__all__ = [
    "AssistantMessageWidget",
    "ChatTranscript",
    "PlainMessageWidget",
]
