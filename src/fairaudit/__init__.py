"""fairaudit: a group-fairness audit toolkit and service."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fairaudit")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0"

from fairaudit.audit import ColumnMapping, audit, audit_dataframe
from fairaudit.errors import AuditInputError
from fairaudit.report import AuditPolicy, AuditReport

__all__ = [
    "AuditInputError",
    "AuditPolicy",
    "AuditReport",
    "ColumnMapping",
    "__version__",
    "audit",
    "audit_dataframe",
]
