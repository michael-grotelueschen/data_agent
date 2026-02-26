"""CLI entry point for the data analyst agent."""

import os
import sys
import subprocess
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    name="data-analyst",
    help="Analyze CSV files using natural language queries powered by Claude."
)
console = Console()


def validate_csv_path(csv_path: str) -> Path:
    """Validate that the CSV file exists and is readable."""
    path = Path(csv_path)

    if not path.exists():
        console.print(f"[red]Error: File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    if not path.is_file():
        console.print(f"[red]Error: Not a file: {csv_path}[/red]")
        raise typer.Exit(1)

    if path.suffix.lower() != ".csv":
        console.print(f"[yellow]Warning: File does not have .csv extension: {csv_path}[/yellow]")

    return path


def check_api_key():
    """Check that ANTHROPIC_API_KEY is set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]Error: ANTHROPIC_API_KEY environment variable is not set.[/red]\n"
            "Please set it with: export ANTHROPIC_API_KEY=your-api-key"
        )
        raise typer.Exit(1)


@app.command()
def analyze(
    csv_path: str = typer.Argument(
        ...,
        help="Path to the CSV file to analyze"
    ),
    query: str = typer.Argument(
        ...,
        help="Natural language query describing what analysis to perform"
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations", "-m",
        help="Maximum number of agent iterations"
    )
):
    """Analyze a CSV file using natural language queries (CLI mode).

    Examples:
        python main.py analyze data.csv "Show me the first 10 rows"
        python main.py analyze sales.csv "What is the average order value by month?"
    """
    from agent import DataAnalystAgent

    check_api_key()
    csv_file = validate_csv_path(csv_path)

    console.print(f"\n[bold cyan]Data Analyst Agent[/bold cyan]")
    console.print(f"[dim]Analyzing: {csv_file}[/dim]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    try:
        agent = DataAnalystAgent(
            csv_path=str(csv_file),
            max_iterations=max_iterations
        )
        agent.run(query)

    except FileNotFoundError as e:
        console.print(f"[red]Error reading CSV file: {e}[/red]")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user.[/yellow]")
        raise typer.Exit(130)


@app.command()
def serve(
    port: int = typer.Option(
        8501,
        "--port", "-p",
        help="Port to run the server on"
    )
):
    """Start the web UI for the Data Analyst Agent.

    Examples:
        python main.py serve
        python main.py serve --port 8080
    """
    check_api_key()

    console.print(f"\n[bold cyan]Data Analyst Agent - Web UI[/bold cyan]")
    console.print(f"[dim]Starting Streamlit server at http://localhost:{port}[/dim]\n")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_path = script_dir / "app.py"

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        f"--server.port={port}",
        "--server.address=localhost"
    ])


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Data Analyst Agent - Analyze CSV files with natural language.

    Use 'analyze' for CLI mode or 'serve' to start the web UI.
    """
    if ctx.invoked_subcommand is None:
        console.print("[dim]No command specified. Starting web UI...[/dim]\n")
        check_api_key()

        script_dir = Path(__file__).parent
        app_path = script_dir / "app.py"

        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost"
        ])


if __name__ == "__main__":
    app()
