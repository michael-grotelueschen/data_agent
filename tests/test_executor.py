"""Tests for the code executor module."""

import os
import tempfile
import pytest
import pandas as pd
from pathlib import Path

from executor import CodeExecutor, ExecutionResult


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "salary": [50000, 60000, 70000]
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def executor(sample_csv, tmp_path):
    """Create a CodeExecutor instance with test data."""
    output_dir = tmp_path / "output"
    return CodeExecutor(sample_csv, output_dir=str(output_dir))


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_successful_result(self):
        result = ExecutionResult(success=True, output="Hello")
        assert result.success is True
        assert result.output == "Hello"
        assert result.error is None
        assert result.figures == []

    def test_failed_result(self):
        result = ExecutionResult(success=False, output="", error="SyntaxError")
        assert result.success is False
        assert result.error == "SyntaxError"

    def test_result_with_figures(self):
        result = ExecutionResult(
            success=True,
            output="",
            figures=["fig1.png", "fig2.png"]
        )
        assert len(result.figures) == 2


class TestCodeExecutor:
    """Tests for the CodeExecutor class."""

    def test_init_loads_dataframe(self, executor):
        """Test that executor loads the CSV into a DataFrame."""
        assert executor.df is not None
        assert len(executor.df) == 3
        assert list(executor.df.columns) == ["name", "age", "salary"]

    def test_init_creates_output_dir(self, executor):
        """Test that executor creates the output directory."""
        assert Path(executor.output_dir).exists()

    def test_execute_simple_code(self, executor):
        """Test executing simple print statement."""
        result = executor.execute("print('Hello, World!')")
        assert result.success is True
        assert "Hello, World!" in result.output
        assert result.error is None

    def test_execute_dataframe_operations(self, executor):
        """Test that df is available in execution context."""
        result = executor.execute("print(df.shape)")
        assert result.success is True
        assert "(3, 3)" in result.output

    def test_execute_pandas_operations(self, executor):
        """Test pandas operations on the dataframe."""
        result = executor.execute("print(df['age'].mean())")
        assert result.success is True
        assert "30.0" in result.output

    def test_execute_with_numpy(self, executor):
        """Test that numpy is available."""
        result = executor.execute("print(np.array([1, 2, 3]).sum())")
        assert result.success is True
        assert "6" in result.output

    def test_execute_syntax_error(self, executor):
        """Test handling of syntax errors."""
        result = executor.execute("print('unclosed string")
        assert result.success is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_execute_runtime_error(self, executor):
        """Test handling of runtime errors."""
        result = executor.execute("x = 1 / 0")
        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    def test_execute_name_error(self, executor):
        """Test handling of undefined variable errors."""
        result = executor.execute("print(undefined_variable)")
        assert result.success is False
        assert "NameError" in result.error

    def test_execute_key_error(self, executor):
        """Test handling of KeyError for non-existent columns."""
        result = executor.execute("print(df['nonexistent_column'])")
        assert result.success is False
        assert "KeyError" in result.error

    def test_execute_multiline_code(self, executor):
        """Test executing multiline code."""
        code = """
total = 0
for x in range(5):
    total += x
print(total)
"""
        result = executor.execute(code)
        assert result.success is True
        assert "10" in result.output

    def test_execute_creates_figure(self, executor):
        """Test that matplotlib figures are saved."""
        code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig(f'{output_dir}/test_plot.png')
plt.close()
"""
        result = executor.execute(code)
        assert result.success is True
        assert len(result.figures) >= 1
        assert any("test_plot.png" in f for f in result.figures)

    def test_execute_auto_saves_unsaved_figures(self, executor):
        """Test that unsaved figures are automatically saved."""
        code = """
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Test Plot')
"""
        result = executor.execute(code)
        assert result.success is True
        assert len(result.figures) >= 1

    def test_execute_seaborn_available(self, executor):
        """Test that seaborn is available."""
        code = """
sns.set_theme()
print('seaborn loaded')
"""
        result = executor.execute(code)
        assert result.success is True
        assert "seaborn loaded" in result.output

    def test_execute_df_not_mutated(self, executor):
        """Test that original dataframe is not mutated."""
        original_len = len(executor.df)
        code = """
df['new_col'] = df['age'] * 2
df = df.drop(0)
print(len(df))
"""
        result = executor.execute(code)
        assert result.success is True
        # Original dataframe should be unchanged
        assert len(executor.df) == original_len
        assert "new_col" not in executor.df.columns

    def test_execute_captures_stderr(self, executor):
        """Test that stderr is captured."""
        code = """
import sys
print('stdout message')
print('stderr message', file=sys.stderr)
"""
        result = executor.execute(code)
        assert result.success is True
        assert "stdout message" in result.output

    def test_get_dataframe_info(self, executor):
        """Test getting dataframe information."""
        info = executor.get_dataframe_info()
        assert "name" in info
        assert "age" in info
        assert "salary" in info
        assert "3 rows" in info or "(3," in info

    def test_execute_empty_code(self, executor):
        """Test executing empty code."""
        result = executor.execute("")
        assert result.success is True
        assert result.output == ""

    def test_execute_only_comments(self, executor):
        """Test executing code with only comments."""
        result = executor.execute("# This is a comment")
        assert result.success is True


class TestCodeExecutorEdgeCases:
    """Edge case tests for CodeExecutor."""

    def test_invalid_csv_path(self, tmp_path):
        """Test handling of invalid CSV path."""
        with pytest.raises(FileNotFoundError):
            CodeExecutor("/nonexistent/path.csv")

    def test_empty_csv(self, tmp_path):
        """Test handling of empty CSV file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("col1,col2,col3\n")
        executor = CodeExecutor(str(csv_path), output_dir=str(tmp_path / "output"))
        assert len(executor.df) == 0
        assert list(executor.df.columns) == ["col1", "col2", "col3"]

    def test_csv_with_special_characters(self, tmp_path):
        """Test CSV with special characters in data."""
        csv_path = tmp_path / "special.csv"
        df = pd.DataFrame({
            "text": ["Hello, World", "Line1\nLine2", "Quote\"Here"],
            "value": [1, 2, 3]
        })
        df.to_csv(csv_path, index=False)
        executor = CodeExecutor(str(csv_path), output_dir=str(tmp_path / "output"))
        result = executor.execute("print(len(df))")
        assert result.success is True
        assert "3" in result.output

    def test_multiple_figures_in_one_execution(self, executor):
        """Test creating multiple figures in one execution."""
        code = """
plt.figure()
plt.plot([1, 2], [1, 2])
plt.savefig(f'{output_dir}/fig1.png')
plt.close()

plt.figure()
plt.plot([3, 4], [3, 4])
plt.savefig(f'{output_dir}/fig2.png')
plt.close()
"""
        result = executor.execute(code)
        assert result.success is True
        assert len(result.figures) >= 2
