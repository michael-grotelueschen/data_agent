"""Output formatting and display using rich."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.prompt import Prompt

from executor import ExecutionResult


class ResultPresenter:
    """Handles formatting and display of results to the terminal."""

    def __init__(self):
        self.console = Console()

    def show_thinking(self, message: str):
        """Show agent's current action/thinking."""
        self.console.print(f"[dim italic]{message}[/dim italic]")

    def show_code_execution(self, code: str, purpose: str, result: ExecutionResult):
        """Display code being run and its output.

        Args:
            code: The Python code that was executed
            purpose: Brief explanation of what the code does
            result: The execution result
        """
        # Show purpose
        self.console.print(f"\n[bold blue]Executing:[/bold blue] {purpose}")

        # Show the code with syntax highlighting
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="Code", border_style="blue"))

        # Show output
        if result.success:
            if result.output.strip():
                self.console.print(
                    Panel(
                        result.output.strip(),
                        title="Output",
                        border_style="green"
                    )
                )
            else:
                self.console.print("[dim]Code executed successfully (no output)[/dim]")

            # Show generated figures
            if result.figures:
                self.console.print(f"\n[bold green]Generated {len(result.figures)} visualization(s):[/bold green]")
                for fig_path in result.figures:
                    self.console.print(f"  [cyan]📊 {fig_path}[/cyan]")
        else:
            self.console.print(
                Panel(
                    result.error or "Unknown error",
                    title="Error",
                    border_style="red"
                )
            )

    def show_question(self, question: str) -> str:
        """Display clarifying question and get user input.

        Args:
            question: The question to ask the user

        Returns:
            User's response
        """
        self.console.print()
        self.console.print(
            Panel(
                question,
                title="Clarifying Question",
                border_style="yellow"
            )
        )
        response = Prompt.ask("[bold yellow]Your answer[/bold yellow]")
        return response

    def show_results(self, summary: str, figures: list[str] | None = None):
        """Display final analysis results.

        Args:
            summary: Markdown-formatted summary of findings
            figures: Optional list of visualization file paths
        """
        self.console.print()
        self.console.print("[bold green]═══ Analysis Results ═══[/bold green]")
        self.console.print()

        # Render markdown summary
        md = Markdown(summary)
        self.console.print(md)

        # Show figures if any
        if figures:
            self.console.print()
            self.console.print("[bold]Visualizations:[/bold]")
            for fig_path in figures:
                self.console.print(f"  [cyan]📊 {fig_path}[/cyan]")

        self.console.print()
        self.console.print("[bold green]═══════════════════════[/bold green]")

    def show_error(self, message: str):
        """Display an error message.

        Args:
            message: The error message to display
        """
        self.console.print(
            Panel(
                message,
                title="Error",
                border_style="red"
            )
        )

    def show_warning(self, message: str):
        """Display a warning message.

        Args:
            message: The warning message to display
        """
        self.console.print(f"[yellow]⚠️  {message}[/yellow]")

    def show_info(self, message: str):
        """Display an info message.

        Args:
            message: The info message to display
        """
        self.console.print(f"[blue]ℹ️  {message}[/blue]")

    def show_dataframe_info(self, info: str):
        """Display information about the loaded DataFrame.

        Args:
            info: String containing DataFrame information
        """
        self.console.print(
            Panel(
                info,
                title="Loaded Data",
                border_style="cyan"
            )
        )
