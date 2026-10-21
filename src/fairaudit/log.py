"""Logging configuration shared by the CLI and the API."""

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once; later calls only adjust the level."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format=_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
    root.setLevel(level.upper() if isinstance(level, str) else level)
