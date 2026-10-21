"""Exceptions raised by fairaudit."""


class AuditInputError(ValueError):
    """Invalid audit input (bad column, non-binary label, mismatched lengths…).

    The API maps it to HTTP 422; the CLI prints it and exits with a non-zero code.
    """
