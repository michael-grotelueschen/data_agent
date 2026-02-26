"""Tests for the agent module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from agent import DataAnalystAgent
from executor import ExecutionResult


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
def mock_llm():
    """Create a mock LLM client."""
    with patch('agent.LLMClient') as mock:
        yield mock


@pytest.fixture
def mock_presenter():
    """Create a mock presenter."""
    with patch('agent.ResultPresenter') as mock:
        instance = mock.return_value
        instance.show_thinking = Mock()
        instance.show_code_execution = Mock()
        instance.show_question = Mock(return_value="user response")
        instance.show_results = Mock()
        instance.show_warning = Mock()
        yield instance


@pytest.fixture
def agent(sample_csv, mock_llm, mock_presenter):
    """Create a DataAnalystAgent with mocked dependencies."""
    return DataAnalystAgent(sample_csv, max_iterations=5)


class TestDataAnalystAgentInit:
    """Tests for DataAnalystAgent initialization."""

    def test_init_sets_csv_path(self, sample_csv, mock_llm, mock_presenter):
        """Test that init sets the CSV path."""
        agent = DataAnalystAgent(sample_csv)
        assert agent.csv_path == sample_csv

    def test_init_sets_max_iterations(self, sample_csv, mock_llm, mock_presenter):
        """Test that init sets max iterations."""
        agent = DataAnalystAgent(sample_csv, max_iterations=20)
        assert agent.max_iterations == 20

    def test_init_empty_messages(self, sample_csv, mock_llm, mock_presenter):
        """Test that init starts with empty messages."""
        agent = DataAnalystAgent(sample_csv)
        assert agent.messages == []

    def test_init_empty_figures(self, sample_csv, mock_llm, mock_presenter):
        """Test that init starts with empty figures list."""
        agent = DataAnalystAgent(sample_csv)
        assert agent.figures == []

    def test_init_creates_executor(self, sample_csv, mock_llm, mock_presenter):
        """Test that init creates executor."""
        with patch('agent.CodeExecutor') as mock_executor:
            agent = DataAnalystAgent(sample_csv)
            mock_executor.assert_called_once_with(sample_csv)


class TestHandleToolCall:
    """Tests for _handle_tool_call method."""

    def test_handle_execute_code(self, agent):
        """Test handling execute_code tool call."""
        with patch.object(agent, '_handle_execute_code', return_value="Output: 42") as mock:
            result = agent._handle_tool_call(
                "execute_code",
                {"code": "print(42)", "purpose": "test"}
            )
            mock.assert_called_once_with({"code": "print(42)", "purpose": "test"})
            assert result == "Output: 42"

    def test_handle_ask_question(self, agent):
        """Test handling ask_clarifying_question tool call."""
        with patch.object(agent, '_handle_ask_question', return_value="User response: yes") as mock:
            result = agent._handle_tool_call(
                "ask_clarifying_question",
                {"question": "Which column?"}
            )
            mock.assert_called_once()
            assert "User response" in result or result == "User response: yes"

    def test_handle_present_results(self, agent):
        """Test handling present_results tool call."""
        with patch.object(agent, '_handle_present_results', return_value="Results presented") as mock:
            result = agent._handle_tool_call(
                "present_results",
                {"summary": "Analysis done"}
            )
            mock.assert_called_once()

    def test_handle_unknown_tool(self, agent):
        """Test handling unknown tool."""
        result = agent._handle_tool_call("unknown_tool", {})
        assert "Error" in result
        assert "unknown_tool" in result


class TestHandleExecuteCode:
    """Tests for _handle_execute_code method."""

    def test_execute_code_success(self, agent):
        """Test successful code execution."""
        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True,
            output="42",
            error=None,
            figures=[]
        ))

        result = agent._handle_execute_code({
            "code": "print(42)",
            "purpose": "Print number"
        })

        assert "42" in result
        agent.presenter.show_code_execution.assert_called_once()

    def test_execute_code_with_figures(self, agent):
        """Test code execution that generates figures."""
        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True,
            output="",
            error=None,
            figures=["output/chart.png"]
        ))

        result = agent._handle_execute_code({
            "code": "plt.plot([1,2,3])",
            "purpose": "Create plot"
        })

        assert "chart.png" in result
        assert "output/chart.png" in agent.figures

    def test_execute_code_failure(self, agent):
        """Test failed code execution."""
        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=False,
            output="",
            error="NameError: undefined",
            figures=[]
        ))

        result = agent._handle_execute_code({
            "code": "print(undefined)",
            "purpose": "Test error"
        })

        assert "Error" in result
        assert "NameError" in result

    def test_execute_code_no_output(self, agent):
        """Test code execution with no output."""
        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True,
            output="",
            error=None,
            figures=[]
        ))

        result = agent._handle_execute_code({
            "code": "x = 1",
            "purpose": "Assign"
        })

        assert "success" in result.lower() or "no output" in result.lower()


class TestHandleAskQuestion:
    """Tests for _handle_ask_question method."""

    def test_ask_question_returns_response(self, agent):
        """Test that ask_question returns user response."""
        agent.presenter.show_question = Mock(return_value="column_a")

        result = agent._handle_ask_question({"question": "Which column?"})

        agent.presenter.show_question.assert_called_once_with("Which column?")
        assert "column_a" in result

    def test_ask_question_formats_response(self, agent):
        """Test that response is formatted correctly."""
        agent.presenter.show_question = Mock(return_value="yes")

        result = agent._handle_ask_question({"question": "Continue?"})

        assert "User response" in result


class TestHandlePresentResults:
    """Tests for _handle_present_results method."""

    def test_present_results_shows_summary(self, agent):
        """Test that present_results shows the summary."""
        result = agent._handle_present_results({
            "summary": "Analysis complete"
        })

        agent.presenter.show_results.assert_called_once()
        assert "success" in result.lower()

    def test_present_results_with_visualizations(self, agent):
        """Test present_results with visualizations."""
        agent.figures = ["output/existing.png"]

        result = agent._handle_present_results({
            "summary": "Results",
            "visualizations": ["output/new.png"]
        })

        call_args = agent.presenter.show_results.call_args
        figures_arg = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get('figures', [])
        # Should combine both existing and new figures
        assert len(figures_arg) >= 1

    def test_present_results_no_visualizations(self, agent):
        """Test present_results without visualizations."""
        result = agent._handle_present_results({
            "summary": "No charts needed"
        })

        agent.presenter.show_results.assert_called_once()


class TestAgentRun:
    """Tests for the run method."""

    def test_run_creates_initial_message(self, agent):
        """Test that run creates an initial message with context."""
        # Create a response that ends the conversation
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        agent.llm.send_message = Mock(return_value=mock_response)
        agent.llm.extract_tool_use = Mock(return_value=[])
        agent.llm.extract_text = Mock(return_value="Done")

        agent.run("Show me the data")

        # Check that initial message was created
        assert len(agent.messages) >= 1
        initial_msg = agent.messages[0]
        assert initial_msg["role"] == "user"
        assert "Show me the data" in initial_msg["content"]

    def test_run_calls_llm(self, agent):
        """Test that run calls the LLM."""
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        agent.llm.send_message = Mock(return_value=mock_response)
        agent.llm.extract_tool_use = Mock(return_value=[])
        agent.llm.extract_text = Mock(return_value="Analysis complete")

        agent.run("Analyze the data")

        agent.llm.send_message.assert_called()

    def test_run_processes_tool_calls(self, agent):
        """Test that run processes tool calls."""
        # First response: tool call
        tool_response = Mock()
        tool_response.content = [Mock(type="tool_use", id="1", name="execute_code", input={"code": "print(1)", "purpose": "test"})]
        tool_response.stop_reason = "tool_use"

        # Second response: present results
        final_response = Mock()
        present_block = Mock()
        present_block.type = "tool_use"
        present_block.id = "2"
        present_block.name = "present_results"
        present_block.input = {"summary": "Done"}
        final_response.content = [present_block]
        final_response.stop_reason = "end_turn"

        agent.llm.send_message = Mock(side_effect=[tool_response, final_response])
        agent.llm.extract_tool_use = Mock(side_effect=[
            [("1", "execute_code", {"code": "print(1)", "purpose": "test"})],
            [("2", "present_results", {"summary": "Done"})]
        ])
        agent.llm.extract_text = Mock(return_value="")
        agent.llm.format_tool_result = Mock(return_value={"type": "tool_result", "tool_use_id": "1", "content": "Output: 1"})

        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True, output="1", error=None, figures=[]
        ))

        result = agent.run("Calculate something")

        assert "Done" in result

    def test_run_stops_at_max_iterations(self, agent):
        """Test that run stops at max iterations."""
        agent.max_iterations = 2

        # Always return a tool call
        mock_response = Mock()
        mock_response.content = [Mock(type="tool_use")]
        mock_response.stop_reason = "tool_use"

        agent.llm.send_message = Mock(return_value=mock_response)
        agent.llm.extract_tool_use = Mock(return_value=[
            ("1", "execute_code", {"code": "print(1)", "purpose": "test"})
        ])
        agent.llm.extract_text = Mock(return_value="")
        agent.llm.format_tool_result = Mock(return_value={"type": "tool_result"})

        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True, output="1", error=None, figures=[]
        ))

        result = agent.run("Keep going forever")

        # Should have called send_message max_iterations times
        assert agent.llm.send_message.call_count <= agent.max_iterations
        agent.presenter.show_warning.assert_called()

    def test_run_handles_api_error(self, agent):
        """Test that run handles API errors gracefully."""
        agent.llm.send_message = Mock(side_effect=Exception("API Error"))

        result = agent.run("This will fail")

        agent.presenter.show_error.assert_called()

    def test_run_ends_on_present_results(self, agent):
        """Test that run ends when present_results is called."""
        mock_response = Mock()
        present_block = Mock()
        present_block.type = "tool_use"
        present_block.id = "1"
        present_block.name = "present_results"
        present_block.input = {"summary": "Final answer"}
        mock_response.content = [present_block]
        mock_response.stop_reason = "tool_use"

        agent.llm.send_message = Mock(return_value=mock_response)
        agent.llm.extract_tool_use = Mock(return_value=[
            ("1", "present_results", {"summary": "Final answer"})
        ])
        agent.llm.extract_text = Mock(return_value="")
        agent.llm.format_tool_result = Mock(return_value={"type": "tool_result"})

        result = agent.run("Quick analysis")

        assert "Final answer" in result
        # Should only call send_message once
        assert agent.llm.send_message.call_count == 1


class TestAgentConversationFlow:
    """Tests for conversation message flow."""

    def test_messages_include_context(self, agent):
        """Test that messages include DataFrame context."""
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"

        agent.llm.send_message = Mock(return_value=mock_response)
        agent.llm.extract_tool_use = Mock(return_value=[])
        agent.llm.extract_text = Mock(return_value="Done")

        agent.run("Analyze")

        initial_message = agent.messages[0]["content"]
        # Should contain dataframe info
        assert "name" in initial_message or "DataFrame" in initial_message

    def test_tool_results_added_to_messages(self, agent):
        """Test that tool results are added to conversation."""
        # First call: execute code
        exec_response = Mock()
        exec_response.content = [Mock(type="tool_use")]
        exec_response.stop_reason = "tool_use"

        # Second call: end
        end_response = Mock()
        end_response.content = []
        end_response.stop_reason = "end_turn"

        agent.llm.send_message = Mock(side_effect=[exec_response, end_response])
        agent.llm.extract_tool_use = Mock(side_effect=[
            [("1", "execute_code", {"code": "print(1)", "purpose": "test"})],
            []
        ])
        agent.llm.extract_text = Mock(return_value="Done")
        agent.llm.format_tool_result = Mock(return_value={
            "type": "tool_result",
            "tool_use_id": "1",
            "content": "Output: 1",
            "is_error": False
        })

        agent.executor.execute = Mock(return_value=ExecutionResult(
            success=True, output="1", error=None, figures=[]
        ))

        agent.run("Run code")

        # Messages should include: initial user message, assistant response, tool result
        assert len(agent.messages) >= 3
        # Last message should be tool result
        tool_result_msg = agent.messages[-1]
        assert tool_result_msg["role"] == "user"
