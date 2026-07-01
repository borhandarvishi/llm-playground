# Prompt Forge

A minimal Gradio app for running templated LLM prompts across **OpenAI** and **Gemini**.

Write a prompt, fill `{{variables}}`, pick a model, and send. Each run shows the response and estimated cost.

## Features

- OpenAI and Gemini support
- Prompt from text or file (`.txt`, `.md`, `.py`)
- Auto-detect and fill `{{variable_name}}` placeholders
- Model and temperature controls
- Per-request token usage and cost estimate
- Run history

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key
GEMINI_API_KEY=your-key
OPENAI_MODEL=gpt-5.4
GEMINI_MODEL=gemini-2.5-flash
TEMPERATURE=0.2
```

## Run

```bash
python llm_chat_app.py
```

Open **http://127.0.0.1:7860**

## Example prompt

```text
Write a short reply to this post:

Title: {{post_title}}
Body: {{post_body}}
```

Variables appear in the middle column. Fill them in, then click **Send**.

## Project layout

```
llm_chat_app.py    # Gradio UI
llm_client.py       # OpenAI / Gemini API calls
prompt_utils.py     # Variable parsing and substitution
cost_calculator.py  # Token cost estimates
```
