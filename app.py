"""Streamlit app for the data analyst agent."""

import logging
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from executor import CodeExecutor, ExecutionResult
from llm import LLMClient

# Logger (uses config from llm.py)
logger = logging.getLogger(__name__)

# Limits
MAX_FILE_SIZE_MB = 10
MAX_QUERIES_PER_HOUR = 20
RATE_LIMIT_WINDOW_SECONDS = 3600

# Page config
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide"
)

# Check API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY environment variable is not set.")
    st.stop()

# Initialize rate limiting state
if "query_timestamps" not in st.session_state:
    st.session_state.query_timestamps = []


def check_rate_limit() -> bool:
    """Check if user is within rate limit. Returns True if allowed."""
    now = time.time()
    # Remove timestamps outside the window
    st.session_state.query_timestamps = [
        ts for ts in st.session_state.query_timestamps
        if now - ts < RATE_LIMIT_WINDOW_SECONDS
    ]
    return len(st.session_state.query_timestamps) < MAX_QUERIES_PER_HOUR


def record_query():
    """Record a query timestamp for rate limiting."""
    st.session_state.query_timestamps.append(time.time())


def get_file_size_mb(file) -> float:
    """Get file size in MB."""
    file.seek(0, 2)  # Seek to end
    size = file.tell() / (1024 * 1024)
    file.seek(0)  # Reset to beginning
    return size


# Title
st.title("📊 Data Analyst Agent")
st.markdown("Upload a CSV file and ask questions in natural language.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# File size check
if uploaded_file:
    file_size = get_file_size_mb(uploaded_file)
    if file_size > MAX_FILE_SIZE_MB:
        st.error(f"File too large: {file_size:.1f} MB (max {MAX_FILE_SIZE_MB} MB)")
        uploaded_file = None

# Query input
query = st.text_input(
    "What would you like to know about your data?",
    placeholder="e.g., What are the top 5 products by revenue?"
)

# Analyze button
if uploaded_file and query:
    if st.button("Analyze", type="primary"):
        # Check rate limit
        if not check_rate_limit():
            remaining = len(st.session_state.query_timestamps)
            st.error(f"Rate limit reached ({MAX_QUERIES_PER_HOUR} queries/hour). Try again later.")
            st.stop()

        # Record this query
        record_query()
        logger.info(f"New query: '{query}' on file: {uploaded_file.name}")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            csv_path = tmp.name

        try:
            # Initialize components
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)

            executor = CodeExecutor(csv_path, output_dir=str(output_dir))
            llm = LLMClient()

            # Show data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(executor.df.head(10))
                st.caption(f"{len(executor.df):,} rows × {len(executor.df.columns)} columns")

            # Get DataFrame info for context
            df_info = executor.get_dataframe_info()

            # Create initial message
            initial_message = f"""I have loaded a CSV file for analysis. Here is information about the data:

{df_info}

User query: {query}

Please analyze this data to answer the user's question. Start by understanding the data, then write and execute code to perform the analysis. When done, use present_results to show your findings."""

            messages = [{"role": "user", "content": initial_message}]
            figures = []

            # Agent loop
            max_iterations = 10
            status = st.status("Analyzing...", expanded=True)

            for iteration in range(max_iterations):
                status.update(label=f"Analyzing... (step {iteration + 1})")

                # Get response from Claude
                response = llm.send_message(messages)

                # Extract text content
                text_content = llm.extract_text(response)
                if text_content:
                    status.write(text_content)

                # Check if done
                tool_uses = llm.extract_tool_use(response)
                if not tool_uses and response.stop_reason == "end_turn":
                    break

                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Process tool calls
                tool_results = []

                for tool_use_id, tool_name, tool_input in tool_uses:
                    if tool_name == "execute_code":
                        code = tool_input.get("code", "")
                        purpose = tool_input.get("purpose", "Execute code")

                        status.write(f"**Executing:** {purpose}")
                        with status.container():
                            st.code(code, language="python")

                        result = executor.execute(code)
                        logger.info(f"Code execution: success={result.success}, figures={len(result.figures)}")

                        if result.success:
                            if result.output:
                                status.code(result.output)
                            if result.figures:
                                figures.extend(result.figures)
                                status.write(f"Generated {len(result.figures)} figure(s)")

                            result_text = f"Output:\n{result.output}" if result.output else "Code executed successfully."
                            if result.figures:
                                result_text += f"\n\nGenerated figures: {', '.join(result.figures)}"
                        else:
                            status.error(result.error)
                            result_text = f"Error:\n{result.error}"

                        tool_results.append(llm.format_tool_result(
                            tool_use_id, result_text, is_error=not result.success
                        ))

                    elif tool_name == "ask_clarifying_question":
                        question = tool_input.get("question", "")
                        status.write(f"**Question:** {question}")
                        tool_results.append(llm.format_tool_result(
                            tool_use_id,
                            "User response: Please proceed with your best judgment."
                        ))

                    elif tool_name == "present_results":
                        summary = tool_input.get("summary", "")
                        viz = tool_input.get("visualizations", [])
                        figures.extend(viz)

                        status.update(label="Analysis complete!", state="complete")
                        logger.info(f"Query complete: {len(figures)} figures generated")

                        # Display results
                        st.markdown("---")
                        st.markdown("## Results")
                        st.markdown(summary)

                        # Display figures
                        if figures:
                            st.markdown("### Visualizations")
                            cols = st.columns(min(len(figures), 2))
                            for i, fig_path in enumerate(set(figures)):
                                if Path(fig_path).exists():
                                    cols[i % 2].image(fig_path)

                        tool_results.append(llm.format_tool_result(
                            tool_use_id, "Results presented to user."
                        ))
                        break

                # Add tool results
                messages.append({"role": "user", "content": tool_results})

            else:
                status.update(label="Reached maximum iterations", state="error")

        except Exception as e:
            st.error(f"Error: {str(e)}")

        finally:
            # Clean up temp file
            try:
                os.unlink(csv_path)
            except:
                pass

elif uploaded_file:
    st.info("Enter a query to analyze your data.")
else:
    st.info("Upload a CSV file to get started.")
