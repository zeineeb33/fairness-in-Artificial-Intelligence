"""Command-line interface: ``fairaudit --help``."""

import typer

from fairaudit import __version__

app = typer.Typer(help="Group-fairness audit toolkit.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Audit binary classifiers for group fairness."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
