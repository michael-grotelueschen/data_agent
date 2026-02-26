"""Agent loop logic for data analysis."""

from typing import Any

from executor import CodeExecutor, ExecutionResult
from llm import LLMClient
from presenter import ResultPresenter


class DataAnalystAgent:
    """Orchestrates the data analysis workflow using Claude."""

    def __init__(self, csv_path: str, max_iterations: int = 10):
        self.csv_path = csv_path
        self.max_iterations = max_iterations
        self.messages: list[dict[str, Any]] = []
        self.figures: list[str] = []

        # Initialize components
        self.executor = CodeExecutor(csv_path)
        self.llm = LLMClient()
        self.presenter = ResultPresenter()

    def run(self, query: str) -> str:
        """Execute the agent loop until completion.

        Args:
            query: The user's natural language query

        Returns:
            The final analysis summary
        """
        # Get DataFrame info for context
        df_info = self.executor.get_dataframe_info()

        # Create initial message with context
        initial_message = f"""I have loaded a CSV file for analysis. Here is information about the data:

{df_info}

User query: {query}

Please analyze this data to answer the user's question. Start by understanding the data, then write and execute code to perform the analysis."""

        self.messages.append({
            "role": "user",
            "content": initial_message
        })

        # Main agent loop
        iteration = 0
        final_summary = None

        while iteration < self.max_iterations:
            iteration += 1
            self.presenter.show_thinking(f"Thinking... (iteration {iteration}/{self.max_iterations})")

            # Get response from Claude
            try:
                response = self.llm.send_message(self.messages)
            except Exception as e:
                self.presenter.show_error(f"API Error: {str(e)}")
                break

            # Extract any text content
            text_content = self.llm.extract_text(response)
            if text_content:
                self.presenter.show_thinking(text_content)

            # Check if we're done (no tool use)
            tool_uses = self.llm.extract_tool_use(response)
            if not tool_uses and response.stop_reason == "end_turn":
                # Claude finished without using a tool
                final_summary = text_content or "Analysis complete."
                break

            # Add assistant's response to messages
            self.messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Process each tool use
            tool_results = []
            for tool_use_id, tool_name, tool_input in tool_uses:
                result = self._handle_tool_call(tool_name, tool_input)

                if tool_name == "present_results":
                    final_summary = tool_input.get("summary", "")
                    tool_results.append(
                        self.llm.format_tool_result(tool_use_id, "Results presented to user.")
                    )
                elif tool_name == "ask_clarifying_question":
                    # User's response becomes the tool result
                    tool_results.append(
                        self.llm.format_tool_result(tool_use_id, result)
                    )
                else:
                    # Code execution result
                    is_error = result.startswith("Error:")
                    tool_results.append(
                        self.llm.format_tool_result(tool_use_id, result, is_error=is_error)
                    )

            # Add tool results to messages
            self.messages.append({
                "role": "user",
                "content": tool_results
            })

            # If we presented results, we're done
            if final_summary is not None:
                break

        # Handle max iterations reached
        if iteration >= self.max_iterations and final_summary is None:
            self.presenter.show_warning(
                f"Reached maximum iterations ({self.max_iterations}). "
                "Presenting partial results."
            )
            final_summary = "Analysis incomplete - maximum iterations reached."

        return final_summary or "No results generated."

    def _handle_tool_call(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Route tool calls to appropriate handlers.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input parameters for the tool

        Returns:
            Result string to send back to Claude
        """
        if tool_name == "execute_code":
            return self._handle_execute_code(tool_input)
        elif tool_name == "ask_clarifying_question":
            return self._handle_ask_question(tool_input)
        elif tool_name == "present_results":
            return self._handle_present_results(tool_input)
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _handle_execute_code(self, tool_input: dict[str, Any]) -> str:
        """Handle code execution tool call.

        Args:
            tool_input: Contains 'code' and 'purpose'

        Returns:
            Execution result as string
        """
        code = tool_input.get("code", "")
        purpose = tool_input.get("purpose", "Execute code")

        result = self.executor.execute(code)

        # Display the execution to user
        self.presenter.show_code_execution(code, purpose, result)

        # Track any generated figures
        if result.figures:
            self.figures.extend(result.figures)

        # Format result for Claude
        if result.success:
            response = f"Output:\n{result.output}" if result.output else "Code executed successfully (no output)."
            if result.figures:
                response += f"\n\nGenerated figures: {', '.join(result.figures)}"
            return response
        else:
            return f"Error:\n{result.error}"

    def _handle_ask_question(self, tool_input: dict[str, Any]) -> str:
        """Handle clarifying question tool call.

        Args:
            tool_input: Contains 'question'

        Returns:
            User's response
        """
        question = tool_input.get("question", "")
        user_response = self.presenter.show_question(question)
        return f"User response: {user_response}"

    def _handle_present_results(self, tool_input: dict[str, Any]) -> str:
        """Handle presenting final results.

        Args:
            tool_input: Contains 'summary' and optional 'visualizations'

        Returns:
            Confirmation string
        """
        summary = tool_input.get("summary", "")
        visualizations = tool_input.get("visualizations", [])

        # Combine specified visualizations with those we tracked
        all_figures = list(set(visualizations + self.figures))

        self.presenter.show_results(summary, all_figures)

        return "Results presented successfully."
