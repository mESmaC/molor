"""Local, vendor-neutral teacher/student evaluation interface."""

from molor.teacher.protocol import RUBRIC, TOOL_SCHEMAS
from molor.teacher.service import StudentConfig, TeacherStudentService

__all__ = ["RUBRIC", "TOOL_SCHEMAS", "StudentConfig", "TeacherStudentService"]
