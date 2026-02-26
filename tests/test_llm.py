"""Tests for the LLM client module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import anthropic

from llm import LLMClient, TOOLS, SYSTEM_PROMPT


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_tools_list_not_empty(self):
        """Test that tools list is defined."""
        assert len(TOOLS) == 3

    def test_execute_code_tool_defined(self):
        """Test execute_code tool is properly defined."""
        tool = next(t for t in TOOLS if t["name"] == "execute_code")
        assert "description" in tool
        assert "input_schema" in tool
        assert "code" in tool["input_schema"]["properties"]
        assert "purpose" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["code", "purpose"]

    def test_ask_clarifying_question_tool_defined(self):
        """Test ask_clarifying_question tool is properly defined."""
        tool = next(t for t in TOOLS if t["name"] == "ask_clarifying_question")
        assert "description" in tool
        assert "question" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["question"]

    def test_present_results_tool_defined(self):
        """Test present_results tool is properly defined."""
        tool = next(t for t in TOOLS if t["name"] == "present_results")
        assert "description" in tool
        assert "summary" in tool["input_schema"]["properties"]
        assert "visualizations" in tool["input_schema"]["properties"]
        assert "summary" in tool["input_schema"]["required"]

    def test_system_prompt_exists(self):
        """Test that system prompt is defined."""
        assert len(SYSTEM_PROMPT) > 0
        assert "data analyst" in SYSTEM_PROMPT.lower()


class TestLLMClient:
    """Tests for the LLMClient class."""

    @pytest.fixture
    def mock_anthropic(self):
        """Create a mock Anthropic client."""
        with patch('llm.anthropic.Anthropic') as mock:
            yield mock

    @pytest.fixture
    def client(self, mock_anthropic):
        """Create an LLMClient with mocked Anthropic client."""
        return LLMClient()

    def test_init_creates_client(self, mock_anthropic):
        """Test that LLMClient initializes Anthropic client."""
        client = LLMClient()
        mock_anthropic.assert_called_once()

    def test_init_with_custom_model(self, mock_anthropic):
        """Test initialization with custom model."""
        client = LLMClient(model="claude-3-opus")
        assert client.model == "claude-3-opus"

    def test_init_with_custom_retries(self, mock_anthropic):
        """Test initialization with custom retry count."""
        client = LLMClient(max_retries=5)
        assert client.max_retries == 5

    def test_send_message_calls_api(self, client):
        """Test that send_message calls the API correctly."""
        mock_response = Mock()
        mock_response.content = []
        mock_response.stop_reason = "end_turn"
        client.client.messages.create = Mock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        response = client.send_message(messages)

        client.client.messages.create.assert_called_once()
        call_kwargs = client.client.messages.create.call_args.kwargs
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] == TOOLS
        assert "system" in call_kwargs

    def test_send_message_with_custom_system(self, client):
        """Test send_message with custom system prompt."""
        mock_response = Mock()
        mock_response.content = []
        client.client.messages.create = Mock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        custom_system = "Custom system prompt"
        client.send_message(messages, system=custom_system)

        call_kwargs = client.client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == custom_system

    def test_send_message_retries_on_rate_limit(self, client):
        """Test that send_message retries on rate limit errors."""
        mock_response = Mock()
        mock_response.content = []

        client.client.messages.create = Mock(
            side_effect=[
                anthropic.RateLimitError(
                    message="Rate limited",
                    response=Mock(status_code=429),
                    body={}
                ),
                mock_response
            ]
        )

        with patch('llm.time.sleep'):
            messages = [{"role": "user", "content": "Hello"}]
            response = client.send_message(messages)

        assert client.client.messages.create.call_count == 2

    def test_send_message_raises_after_max_retries(self, client):
        """Test that send_message raises after max retries."""
        client.max_retries = 2
        client.client.messages.create = Mock(
            side_effect=anthropic.RateLimitError(
                message="Rate limited",
                response=Mock(status_code=429),
                body={}
            )
        )

        with patch('llm.time.sleep'):
            messages = [{"role": "user", "content": "Hello"}]
            with pytest.raises(anthropic.RateLimitError):
                client.send_message(messages)

        assert client.client.messages.create.call_count == 2


class TestExtractToolUse:
    """Tests for extract_tool_use method."""

    @pytest.fixture
    def client(self):
        """Create an LLMClient with mocked Anthropic client."""
        with patch('llm.anthropic.Anthropic'):
            return LLMClient()

    def test_extract_tool_use_single_tool(self, client):
        """Test extracting a single tool use."""
        mock_block = Mock()
        mock_block.type = "tool_use"
        mock_block.id = "tool_123"
        mock_block.name = "execute_code"
        mock_block.input = {"code": "print('hi')", "purpose": "test"}

        mock_response = Mock()
        mock_response.content = [mock_block]

        result = client.extract_tool_use(mock_response)

        assert len(result) == 1
        assert result[0] == ("tool_123", "execute_code", {"code": "print('hi')", "purpose": "test"})

    def test_extract_tool_use_multiple_tools(self, client):
        """Test extracting multiple tool uses."""
        mock_block1 = Mock()
        mock_block1.type = "tool_use"
        mock_block1.id = "tool_1"
        mock_block1.name = "execute_code"
        mock_block1.input = {"code": "print(1)"}

        mock_block2 = Mock()
        mock_block2.type = "tool_use"
        mock_block2.id = "tool_2"
        mock_block2.name = "present_results"
        mock_block2.input = {"summary": "Done"}

        mock_response = Mock()
        mock_response.content = [mock_block1, mock_block2]

        result = client.extract_tool_use(mock_response)

        assert len(result) == 2
        assert result[0][0] == "tool_1"
        assert result[1][0] == "tool_2"

    def test_extract_tool_use_no_tools(self, client):
        """Test extracting when no tools are used."""
        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hello"

        mock_response = Mock()
        mock_response.content = [mock_block]

        result = client.extract_tool_use(mock_response)

        assert len(result) == 0

    def test_extract_tool_use_mixed_content(self, client):
        """Test extracting from mixed content (text + tool)."""
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Let me help you"

        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.id = "tool_1"
        tool_block.name = "execute_code"
        tool_block.input = {"code": "print(1)"}

        mock_response = Mock()
        mock_response.content = [text_block, tool_block]

        result = client.extract_tool_use(mock_response)

        assert len(result) == 1
        assert result[0][1] == "execute_code"


class TestExtractText:
    """Tests for extract_text method."""

    @pytest.fixture
    def client(self):
        """Create an LLMClient with mocked Anthropic client."""
        with patch('llm.anthropic.Anthropic'):
            return LLMClient()

    def test_extract_text_single_block(self, client):
        """Test extracting text from single text block."""
        mock_block = Mock()
        mock_block.type = "text"
        mock_block.text = "Hello, world!"

        mock_response = Mock()
        mock_response.content = [mock_block]

        result = client.extract_text(mock_response)

        assert result == "Hello, world!"

    def test_extract_text_multiple_blocks(self, client):
        """Test extracting text from multiple text blocks."""
        mock_block1 = Mock()
        mock_block1.type = "text"
        mock_block1.text = "Line 1"

        mock_block2 = Mock()
        mock_block2.type = "text"
        mock_block2.text = "Line 2"

        mock_response = Mock()
        mock_response.content = [mock_block1, mock_block2]

        result = client.extract_text(mock_response)

        assert result == "Line 1\nLine 2"

    def test_extract_text_no_text(self, client):
        """Test extracting when no text blocks present."""
        mock_block = Mock()
        mock_block.type = "tool_use"

        mock_response = Mock()
        mock_response.content = [mock_block]

        result = client.extract_text(mock_response)

        assert result == ""

    def test_extract_text_mixed_content(self, client):
        """Test extracting text from mixed content."""
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Some text"

        tool_block = Mock()
        tool_block.type = "tool_use"

        mock_response = Mock()
        mock_response.content = [text_block, tool_block]

        result = client.extract_text(mock_response)

        assert result == "Some text"


class TestFormatToolResult:
    """Tests for format_tool_result method."""

    @pytest.fixture
    def client(self):
        """Create an LLMClient with mocked Anthropic client."""
        with patch('llm.anthropic.Anthropic'):
            return LLMClient()

    def test_format_tool_result_success(self, client):
        """Test formatting successful tool result."""
        result = client.format_tool_result("tool_123", "Success output")

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_123"
        assert result["content"] == "Success output"
        assert result["is_error"] is False

    def test_format_tool_result_error(self, client):
        """Test formatting error tool result."""
        result = client.format_tool_result("tool_456", "Error message", is_error=True)

        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "tool_456"
        assert result["content"] == "Error message"
        assert result["is_error"] is True
