import typer
from typing import Optional

from mason import __version__
from mason.merge import merge
from mason.chat import chat
from mason.convert import convert_to_gguf


# Initialize the Typer app
app = typer.Typer(
    help="Mason: Model Adaptation & Synthetic Optimization eNgine",
    no_args_is_help=True
)


def version_callback(value: bool):
    """
    Callback function to handle the --version flag.
    """
    if value:
        typer.echo(f"Mason CLI Version: {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        callback=version_callback,
        is_eager=True,
    )
):
    """
    Mason CLI entry point.
    """
    pass


app.command(name="merge")(merge)
app.command(name="chat")(chat)
app.command(name="convert")(convert_to_gguf)


if __name__ == "__main__":
    app()