# Data Agent

An AI-powered data analysis tool that lets you analyze CSV files using natural language queries. Upload a CSV, ask a question, and get insights with visualizations.

## Features

- Natural language queries ("What are the top 5 products by revenue?")
- Automatic Python code generation and execution
- Visualization generation (charts, plots)
- Web UI with drag-and-drop file upload
- CLI mode for scripting

## Requirements

- Python 3.10+
- Anthropic API key

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd data_agent

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=your-api-key
```

## Usage

### Web UI (recommended)

```bash
python main.py serve
```

Open http://localhost:8501, upload a CSV, and ask questions.

### CLI

```bash
python main.py analyze data.csv "What is the average salary by department?"
```

## Limits

- Max file size: 10 MB
- Max queries: 20 per hour
- Max iterations per query: 10

## Project Structure

```
data_agent/
├── app.py           # Streamlit web UI
├── main.py          # CLI entry point
├── agent.py         # Agent loop (CLI mode)
├── executor.py      # Sandboxed code execution
├── llm.py           # Claude API wrapper
├── presenter.py     # CLI output formatting
├── requirements.txt
├── output/          # Generated visualizations
└── tests/           # Test suite
```

## Running Tests

```bash
python -m pytest tests/
```
