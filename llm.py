"""Claude API wrapper and tool definitions."""

import os
import time
from typing import Any

import anthropic


SYSTEM_PROMPT = """You are a data analyst agent. You help users analyze CSV data by writing and executing Python code.

Your workflow:
1. First, understand what the user is asking
2. If the query is unclear, use ask_clarifying_question to get more information
3. Use execute_code to run Python code for analysis. The CSV is pre-loaded as 'df'
4. If code fails, debug and try again
5. Once you have results, use present_results to show findings

Guidelines:
- Always explore the data first (df.head(), df.info(), df.describe())
- Handle missing values appropriately
- Create visualizations when they help explain the data
- Save all figures with plt.savefig('output/figure_name.png')
- Be thorough but concise in your explanations
"""

TOOLS = [
    {
        "name": "execute_code",
        "description": """Execute Python code to analyze the CSV data.
The DataFrame is pre-loaded as 'df'.
Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns).
Always use plt.savefig() to save figures - they will be displayed to the user.
Print any results you want to show.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                },
                "purpose": {
                    "type": "string",
                    "description": "Brief explanation of what this code does"
                }
            },
            "required": ["code", "purpose"]
        }
    },
    {
        "name": "ask_clarifying_question",
        "description": "Ask the user a clarifying question when the query is ambiguous or you need more information to proceed with the analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "present_results",
        "description": "Present the final analysis results to the user. Use this when you have completed the analysis and are ready to show findings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A clear, detailed summary of the findings in markdown format"
                },
                "visualizations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of visualization file paths to display"
                }
            },
            "required": ["summary"]
        }
    }
]


class LLMClient:
    """Wrapper for Claude API interactions."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_retries: int = 3):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_retries = max_retries

    def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None
    ) -> anthropic.types.Message:
        """Send a message to Claude and get a response.

        Args:
            messages: Conversation history
            system: Optional system prompt override

        Returns:
            Claude's response message
        """
        system_prompt = system or SYSTEM_PROMPT

        for attempt in range(self.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages
                )
                return response
            except anthropic.RateLimitError:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise

    def extract_tool_use(
        self,
        response: anthropic.types.Message
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Extract tool use blocks from a response.

        Args:
            response: Claude's response message

        Returns:
            List of (tool_use_id, tool_name, tool_input) tuples
        """
        tool_uses = []
        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append((block.id, block.name, block.input))
        return tool_uses

    def extract_text(self, response: anthropic.types.Message) -> str:
        """Extract text content from a response.

        Args:
            response: Claude's response message

        Returns:
            Concatenated text content
        """
        texts = []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
        return "\n".join(texts)

    def format_tool_result(
        self,
        tool_use_id: str,
        result: str,
        is_error: bool = False
    ) -> dict[str, Any]:
        """Format a tool result for sending back to Claude.

        Args:
            tool_use_id: The ID of the tool use this is responding to
            result: The result content
            is_error: Whether this is an error result

        Returns:
            Formatted tool result message content
        """
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "is_error": is_error
        }
