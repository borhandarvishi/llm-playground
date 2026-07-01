"""
Gradio UI for OpenAI / Gemini chat completions with prompt variables and cost tracking.

Run:
    cd test_ai
    pip install -r requirements.txt
    python llm_chat_app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from cost_calculator import calculate_cost, format_cost_compact_html
from llm_client import (
    GEMINI_MODELS,
    OPENAI_MODELS,
    call_llm,
    model_supports_temperature,
)
from prompt_utils import (
    combine_prompt,
    extract_variables,
    get_file_path,
    substitute_variables,
    values_from_table,
    variables_table_from_prompt,
)

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4").strip()
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding: 10px 14px !important;
}
footer { display: none !important; }

/* lock page to one viewport */
.app-shell {
    height: calc(100vh - 24px);
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow: hidden;
}

/* top toolbar */
.topbar {
    flex: 0 0 auto;
    align-items: center !important;
    gap: 8px !important;
    padding: 8px 10px !important;
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    background: var(--background-fill-secondary);
}
.topbar-title {
    font-size: 0.9rem;
    font-weight: 600;
    margin: 0 !important;
    white-space: nowrap;
    padding-right: 6px;
}
.topbar .file-chip {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 160px;
}

/* three-column workspace */
.workspace {
    flex: 1 1 auto;
    min-height: 0 !important;
    align-items: stretch !important;
    gap: 8px !important;
    overflow: hidden;
}
.pane {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 8px !important;
    min-height: 0 !important;
    overflow: hidden;
    display: flex !important;
    flex-direction: column !important;
}
.pane-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
    margin: 0 0 6px 0 !important;
    flex: 0 0 auto;
}
.pane-body {
    flex: 1 1 auto;
    min-height: 0 !important;
    overflow: auto;
}
.pane-body textarea {
    height: 100% !important;
    min-height: 200px !important;
    resize: none !important;
    font-size: 0.82rem !important;
    line-height: 1.45 !important;
}
.pane-body .table-wrap {
    max-height: 100%;
    overflow: auto;
}
.vars-empty {
    font-size: 0.78rem;
    color: var(--body-text-color-subdued);
    padding: 4px 2px;
}

/* output column */
.cost-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    padding: 6px 8px;
    margin-bottom: 6px;
    border-radius: 6px;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    flex: 0 0 auto;
}
.cost-stat { display: flex; flex-direction: column; gap: 1px; }
.cost-label {
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--body-text-color-subdued);
}
.cost-value {
    font-size: 0.78rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
}
"""


def provider_default_model(provider: str) -> str:
    if provider == "gemini":
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_OPENAI_MODEL


def on_provider_change(provider: str):
    models = GEMINI_MODELS if provider == "gemini" else OPENAI_MODELS
    default_model = provider_default_model(provider)
    choices = models.copy()
    if default_model and default_model not in choices:
        choices.insert(0, default_model)
    supports_temp = model_supports_temperature(default_model)
    return (
        gr.update(choices=choices, value=default_model),
        gr.update(value=DEFAULT_TEMPERATURE, interactive=supports_temp),
    )


def on_model_change(provider: str, preset_model: str):
    model = (preset_model or "").strip() or provider_default_model(provider)
    supports_temp = model_supports_temperature(model)
    return gr.update(value=DEFAULT_TEMPERATURE, interactive=supports_temp)


def refresh_variables(prompt_text: str, prompt_file):
    file_path = get_file_path(prompt_file)
    rows = variables_table_from_prompt(prompt_text, file_path)
    if not rows:
        return gr.update(value=[], visible=False), gr.update(
            value='<div class="vars-empty">No {{variables}} in prompt</div>',
            visible=True,
        )
    return gr.update(value=rows, visible=True), gr.update(visible=False)


def resolve_model(preset_model: str, provider: str) -> str:
    model = (preset_model or "").strip()
    return model or provider_default_model(provider)


def run_llm_request(
    provider: str,
    preset_model: str,
    temperature: float,
    prompt_text: str,
    prompt_file,
    variables_table,
):
    file_path = get_file_path(prompt_file)
    base_prompt = combine_prompt(prompt_text, file_path)

    if not base_prompt:
        return "Enter a prompt or upload a file.", ""

    var_values = values_from_table(variables_table)
    required_vars = extract_variables(base_prompt)
    missing = [name for name in required_vars if not var_values.get(name, "").strip()]
    if missing:
        return f"Missing: {', '.join(missing)}", ""

    final_prompt = substitute_variables(base_prompt, var_values)
    model = resolve_model(preset_model, provider)
    temp = temperature if model_supports_temperature(model) else None

    response = call_llm(
        provider=provider,
        model=model,
        prompt=final_prompt,
        temperature=temp,
    )

    if response.request_error:
        return f"Error: {response.request_error}", ""

    cost = calculate_cost(response)
    return response.text, format_cost_compact_html(response, cost)


def on_upload(prompt_text, uploaded_file):
    file_path = None
    if uploaded_file:
        file_path = uploaded_file if isinstance(uploaded_file, str) else uploaded_file[0]
    vars_table, vars_empty = refresh_variables(prompt_text, file_path)
    chip = ""
    if file_path:
        chip = f'<span class="file-chip">📎 {html.escape(Path(file_path).name)}</span>'
    return file_path, chip, vars_table, vars_empty


def build_ui() -> gr.Blocks:
    initial_provider = DEFAULT_PROVIDER if DEFAULT_PROVIDER in {"openai", "gemini"} else "openai"
    initial_models = GEMINI_MODELS if initial_provider == "gemini" else OPENAI_MODELS
    initial_model = provider_default_model(initial_provider)
    if initial_model not in initial_models:
        initial_models = [initial_model, *initial_models]
    initial_supports_temp = model_supports_temperature(initial_model)

    with gr.Blocks(title="LLM Chat", css=CUSTOM_CSS) as demo:
        prompt_file = gr.State(value=None)

        with gr.Column(elem_classes=["app-shell"]):
            # ── toolbar ──
            with gr.Row(elem_classes=["topbar"]):
                gr.Markdown("LLM Chat", elem_classes=["topbar-title"])
                provider = gr.Radio(
                    choices=["openai", "gemini"],
                    value=initial_provider,
                    label=None,
                    show_label=False,
                    scale=0,
                )
                model_dropdown = gr.Dropdown(
                    choices=initial_models,
                    value=initial_model,
                    label=None,
                    show_label=False,
                    allow_custom_value=True,
                    scale=1,
                    min_width=140,
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=DEFAULT_TEMPERATURE,
                    label="Temp",
                    interactive=initial_supports_temp,
                    scale=1,
                )
                upload_btn = gr.UploadButton(
                    "Upload",
                    file_types=[".txt", ".md", ".py"],
                    file_count="single",
                    size="sm",
                    variant="secondary",
                    scale=0,
                )
                file_chip = gr.HTML(scale=0)
                send_btn = gr.Button("Send", variant="primary", size="sm", scale=0)

            # ── workspace: prompt | variables | output ──
            with gr.Row(equal_height=True, elem_classes=["workspace"]):
                with gr.Column(elem_classes=["pane"], scale=2):
                    gr.Markdown("Prompt", elem_classes=["pane-title"])
                    with gr.Column(elem_classes=["pane-body"]):
                        prompt_text = gr.Textbox(
                            label=None,
                            show_label=False,
                            placeholder="Write or upload a prompt…",
                            lines=20,
                            max_lines=20,
                        )

                with gr.Column(elem_classes=["pane"], scale=1, min_width=200):
                    gr.Markdown("Variables", elem_classes=["pane-title"])
                    with gr.Column(elem_classes=["pane-body"]):
                        vars_empty = gr.HTML(
                            '<div class="vars-empty">No {{variables}} in prompt</div>'
                        )
                        variables_table = gr.Dataframe(
                            headers=["Name", "Value"],
                            datatype=["str", "str"],
                            row_count=(0, "dynamic"),
                            interactive=True,
                            visible=False,
                            label=None,
                            show_label=False,
                            wrap=True,
                        )

                with gr.Column(elem_classes=["pane"], scale=2):
                    gr.Markdown("Output", elem_classes=["pane-title"])
                    cost_bar = gr.HTML("")
                    with gr.Column(elem_classes=["pane-body"]):
                        response_box = gr.Textbox(
                            label=None,
                            show_label=False,
                            lines=20,
                            max_lines=20,
                            interactive=False,
                            placeholder="Response…",
                        )

        provider.change(
            on_provider_change,
            inputs=[provider],
            outputs=[model_dropdown, temperature],
        )
        model_dropdown.change(
            on_model_change,
            inputs=[provider, model_dropdown],
            outputs=[temperature],
        )

        upload_btn.upload(
            on_upload,
            inputs=[prompt_text, upload_btn],
            outputs=[prompt_file, file_chip, variables_table, vars_empty],
        )

        prompt_text.change(
            refresh_variables,
            inputs=[prompt_text, prompt_file],
            outputs=[variables_table, vars_empty],
        )

        send_btn.click(
            run_llm_request,
            inputs=[
                provider,
                model_dropdown,
                temperature,
                prompt_text,
                prompt_file,
                variables_table,
            ],
            outputs=[response_box, cost_bar],
        )

    return demo


def launch_app(demo: gr.Blocks) -> None:
    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": 7860,
        "show_error": True,
        "share": True,
    }
    try:
        theme = gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
            radius_size=gr.themes.sizes.radius_sm,
        ).set(
            button_primary_background_fill="*neutral_800",
            button_primary_background_fill_hover="*neutral_700",
            block_border_width="0px",
        )
        demo.launch(**launch_kwargs, theme=theme, css=CUSTOM_CSS)
    except TypeError:
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app(build_ui())
