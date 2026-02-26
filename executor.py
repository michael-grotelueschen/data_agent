"""Sandboxed code execution for data analysis."""

import io
import os
import sys
import traceback
import threading
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: str | None = None
    figures: list[str] = field(default_factory=list)


class CodeExecutor:
    """Executes Python code in a controlled environment."""

    def __init__(self, csv_path: str, output_dir: str = "output"):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self._ensure_output_dir()
        self.df = pd.read_csv(csv_path)
        self._figure_counter = 0

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_existing_figures(self) -> set[str]:
        """Get set of existing figure files in output directory."""
        output_path = Path(self.output_dir)
        if not output_path.exists():
            return set()
        return {str(f) for f in output_path.glob("*.png")}

    def execute(self, code: str, timeout: int = 30) -> ExecutionResult:
        """Execute code and return results.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds

        Returns:
            ExecutionResult with output, errors, and generated figures
        """
        # Track figures before execution
        figures_before = self._get_existing_figures()

        # Close any existing figures
        plt.close('all')

        # Prepare the execution namespace
        namespace = {
            "df": self.df.copy(),
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "output_dir": self.output_dir,
        }

        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Store result from thread execution
        result_container = {"error": None}

        def run_code():
            try:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    exec(code, namespace)
            except Exception as e:
                result_container["error"] = e

        try:
            # Use thread pool for timeout (works in multi-threaded environments)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_code)
                future.result(timeout=timeout)

            # Check if there was an error during execution
            if result_container["error"]:
                raise result_container["error"]

            # Check for any figures that were created but not saved
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    self._figure_counter += 1
                    fig_path = os.path.join(
                        self.output_dir,
                        f"figure_{self._figure_counter}.png"
                    )
                    plt.figure(fig_num).savefig(fig_path, bbox_inches='tight', dpi=150)
                plt.close('all')

            # Detect new figures
            figures_after = self._get_existing_figures()
            new_figures = sorted(figures_after - figures_before)

            return ExecutionResult(
                success=True,
                output=stdout_buffer.getvalue(),
                error=None,
                figures=new_figures
            )

        except FuturesTimeoutError:
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=f"Timeout: Code execution exceeded {timeout} seconds"
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=error_msg
            )
        finally:
            plt.close('all')

    def get_dataframe_info(self) -> str:
        """Get information about the loaded DataFrame for context."""
        info_buffer = io.StringIO()
        self.df.info(buf=info_buffer)

        info_str = f"""DataFrame loaded from: {self.csv_path}
Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns

Columns and types:
{info_buffer.getvalue()}

Sample data (first 5 rows):
{self.df.head().to_string()}

Basic statistics:
{self.df.describe().to_string()}
"""
        return info_str
