#!/usr/bin/env python3
"""Test script to run the agent and display logging output."""

import os
import sys

# Check API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Error: ANTHROPIC_API_KEY not set")
    print("Run: export ANTHROPIC_API_KEY=your-key")
    sys.exit(1)

# Clear previous log
if os.path.exists("agent.log"):
    os.remove("agent.log")

from executor import CodeExecutor
from llm import LLMClient

def run_query(csv_path: str, query: str):
    """Run a query against a CSV file."""
    print(f"\n{'='*60}")
    print(f"CSV: {csv_path}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    # Initialize
    executor = CodeExecutor(csv_path, output_dir="output")
    llm = LLMClient()

    # Build initial message
    df_info = executor.get_dataframe_info()
    initial_message = f"""I have loaded a CSV file for analysis. Here is information about the data:

{df_info}

User query: {query}

Please analyze this data to answer the user's question. Start by understanding the data, then write and execute code to perform the analysis. When done, use present_results to show your findings."""

    messages = [{"role": "user", "content": initial_message}]
    figures = []

    # Agent loop
    max_iterations = 10
    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Call LLM
        response = llm.send_message(messages)

        # Show thinking
        text_content = llm.extract_text(response)
        if text_content:
            preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
            print(f"Thinking: {preview}")

        # Check if done
        tool_uses = llm.extract_tool_use(response)
        if not tool_uses and response.stop_reason == "end_turn":
            print("Complete (no more tool calls)")
            break

        messages.append({"role": "assistant", "content": response.content})

        # Process tool calls
        tool_results = []
        done = False

        for tool_use_id, tool_name, tool_input in tool_uses:
            if tool_name == "execute_code":
                code = tool_input.get("code", "")
                purpose = tool_input.get("purpose", "Execute code")
                print(f"\nExecuting: {purpose}")

                result = executor.execute(code)

                if result.success:
                    if result.output:
                        preview = result.output[:300] + "..." if len(result.output) > 300 else result.output
                        print(f"Output:\n{preview}")
                    if result.figures:
                        figures.extend(result.figures)
                        print(f"Figures: {result.figures}")
                    result_text = f"Output:\n{result.output}" if result.output else "Code executed successfully."
                    if result.figures:
                        result_text += f"\n\nGenerated figures: {', '.join(result.figures)}"
                else:
                    print(f"Error: {result.error[:200]}")
                    result_text = f"Error:\n{result.error}"

                tool_results.append(llm.format_tool_result(
                    tool_use_id, result_text, is_error=not result.success
                ))

            elif tool_name == "ask_clarifying_question":
                question = tool_input.get("question", "")
                print(f"\nQuestion: {question}")
                tool_results.append(llm.format_tool_result(
                    tool_use_id, "User response: Please proceed with your best judgment."
                ))

            elif tool_name == "present_results":
                summary = tool_input.get("summary", "")
                viz = tool_input.get("visualizations", [])
                figures.extend(viz)

                print(f"\n{'='*60}")
                print("FINAL RESULTS")
                print(f"{'='*60}")
                print(summary)
                print(f"{'='*60}")

                tool_results.append(llm.format_tool_result(
                    tool_use_id, "Results presented to user."
                ))
                done = True

        messages.append({"role": "user", "content": tool_results})

        if done:
            break
    else:
        print("\nWarning: Reached max iterations")

    print(f"\nTotal figures generated: {figures}")
    return figures


if __name__ == "__main__":
    # Default test
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/customer_churn.csv"
    query = sys.argv[2] if len(sys.argv) > 2 else "What explains customer churn?"

    run_query(csv_path, query)

    # Show log
    print(f"\n{'='*60}")
    print("AGENT LOG (agent.log)")
    print(f"{'='*60}")
    if os.path.exists("agent.log"):
        with open("agent.log") as f:
            print(f.read())
