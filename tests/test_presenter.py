"""Tests for the presenter module."""

import pytest
import re
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from rich.console import Console

from presenter import ResultPresenter
from executor import ExecutionResult


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_pattern.sub('', text)


@pytest.fixture
def presenter():
    """Create a ResultPresenter instance."""
    return ResultPresenter()


@pytest.fixture
def capture_console():
    """Create a presenter with captured output."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=80)
    presenter = ResultPresenter()
    presenter.console = console
    return presenter, output


def get_output(output: StringIO) -> str:
    """Get output text with ANSI codes stripped."""
    return strip_ansi(output.getvalue())


class TestResultPresenter:
    """Tests for the ResultPresenter class."""

    def test_init_creates_console(self, presenter):
        """Test that presenter initializes with a console."""
        assert presenter.console is not None
        assert isinstance(presenter.console, Console)


class TestShowThinking:
    """Tests for show_thinking method."""

    def test_show_thinking_outputs_message(self, capture_console):
        """Test that show_thinking outputs the message."""
        presenter, output = capture_console
        presenter.show_thinking("Processing data...")

        output_text = get_output(output)
        assert "Processing data..." in output_text

    def test_show_thinking_with_iteration(self, capture_console):
        """Test show_thinking with iteration info."""
        presenter, output = capture_console
        presenter.show_thinking("Thinking... (iteration 1/10)")

        output_text = get_output(output)
        assert "iteration 1/10" in output_text


class TestShowCodeExecution:
    """Tests for show_code_execution method."""

    def test_show_code_execution_success(self, capture_console):
        """Test displaying successful code execution."""
        presenter, output = capture_console
        result = ExecutionResult(
            success=True,
            output="Result: 42",
            error=None,
            figures=[]
        )

        presenter.show_code_execution(
            code="print('Result:', 42)",
            purpose="Calculate answer",
            result=result
        )

        output_text = output.getvalue()
        assert "Calculate answer" in output_text
        assert "Result: 42" in output_text

    def test_show_code_execution_with_figures(self, capture_console):
        """Test displaying code execution that generated figures."""
        presenter, output = capture_console
        result = ExecutionResult(
            success=True,
            output="",
            error=None,
            figures=["output/chart.png", "output/plot.png"]
        )

        presenter.show_code_execution(
            code="plt.plot([1,2,3])",
            purpose="Create plot",
            result=result
        )

        output_text = output.getvalue()
        assert "2" in output_text or "visualization" in output_text.lower()

    def test_show_code_execution_failure(self, capture_console):
        """Test displaying failed code execution."""
        presenter, output = capture_console
        result = ExecutionResult(
            success=False,
            output="",
            error="NameError: name 'undefined' is not defined",
            figures=[]
        )

        presenter.show_code_execution(
            code="print(undefined)",
            purpose="Test undefined",
            result=result
        )

        output_text = output.getvalue()
        assert "NameError" in output_text

    def test_show_code_execution_no_output(self, capture_console):
        """Test displaying code execution with no output."""
        presenter, output = capture_console
        result = ExecutionResult(
            success=True,
            output="",
            error=None,
            figures=[]
        )

        presenter.show_code_execution(
            code="x = 1",
            purpose="Assign variable",
            result=result
        )

        output_text = output.getvalue()
        assert "success" in output_text.lower() or "no output" in output_text.lower()


class TestShowQuestion:
    """Tests for show_question method."""

    def test_show_question_displays_question(self, capture_console):
        """Test that show_question displays the question."""
        presenter, output = capture_console

        with patch('presenter.Prompt.ask', return_value="User response"):
            response = presenter.show_question("What column should I use?")

        output_text = output.getvalue()
        assert "What column should I use?" in output_text
        assert response == "User response"

    def test_show_question_returns_user_input(self):
        """Test that show_question returns user input."""
        presenter = ResultPresenter()

        with patch('presenter.Prompt.ask', return_value="column_name"):
            response = presenter.show_question("Which column?")

        assert response == "column_name"


class TestShowResults:
    """Tests for show_results method."""

    def test_show_results_displays_summary(self, capture_console):
        """Test that show_results displays the summary."""
        presenter, output = capture_console

        presenter.show_results(
            summary="## Analysis Complete\n\nThe average is 42.",
            figures=[]
        )

        output_text = output.getvalue()
        assert "Analysis" in output_text
        assert "42" in output_text

    def test_show_results_with_figures(self, capture_console):
        """Test show_results with visualizations."""
        presenter, output = capture_console

        presenter.show_results(
            summary="Here are the results",
            figures=["output/chart1.png", "output/chart2.png"]
        )

        output_text = output.getvalue()
        assert "chart1.png" in output_text or "Visualizations" in output_text

    def test_show_results_markdown_formatting(self, capture_console):
        """Test that markdown is rendered."""
        presenter, output = capture_console

        presenter.show_results(
            summary="# Header\n\n- Item 1\n- Item 2\n\n**Bold text**",
            figures=None
        )

        output_text = output.getvalue()
        # Rich should render the markdown
        assert "Header" in output_text or "Item" in output_text

    def test_show_results_none_figures(self, capture_console):
        """Test show_results with None figures."""
        presenter, output = capture_console

        # Should not raise an error
        presenter.show_results(
            summary="Results here",
            figures=None
        )

        output_text = output.getvalue()
        assert "Results" in output_text


class TestShowError:
    """Tests for show_error method."""

    def test_show_error_displays_message(self, capture_console):
        """Test that show_error displays the error message."""
        presenter, output = capture_console

        presenter.show_error("Something went wrong!")

        output_text = output.getvalue()
        assert "Something went wrong!" in output_text

    def test_show_error_formatting(self, capture_console):
        """Test that error is formatted as error."""
        presenter, output = capture_console

        presenter.show_error("File not found")

        output_text = output.getvalue()
        assert "Error" in output_text or "File not found" in output_text


class TestShowWarning:
    """Tests for show_warning method."""

    def test_show_warning_displays_message(self, capture_console):
        """Test that show_warning displays the warning message."""
        presenter, output = capture_console

        presenter.show_warning("This might take a while")

        output_text = output.getvalue()
        assert "This might take a while" in output_text


class TestShowInfo:
    """Tests for show_info method."""

    def test_show_info_displays_message(self, capture_console):
        """Test that show_info displays the info message."""
        presenter, output = capture_console

        presenter.show_info("Loading data...")

        output_text = get_output(output)
        assert "Loading data..." in output_text


class TestShowDataframeInfo:
    """Tests for show_dataframe_info method."""

    def test_show_dataframe_info_displays_info(self, capture_console):
        """Test that show_dataframe_info displays DataFrame info."""
        presenter, output = capture_console

        df_info = """DataFrame: 100 rows x 5 columns
Columns: name, age, salary, department, hire_date"""

        presenter.show_dataframe_info(df_info)

        output_text = output.getvalue()
        assert "100 rows" in output_text or "DataFrame" in output_text
