"""Estimate request cost from token usage and model pricing."""

from __future__ import annotations

from dataclasses import dataclass

from llm_client import LLMResponse


# USD per 1M tokens (input, output). Update as pricing changes.
PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 5.0, "output": 15.0},
    "gpt-5.5": {"input": 5.0, "output": 15.0},
    "gpt-5.2": {"input": 4.0, "output": 12.0},
    "gpt-5.1": {"input": 3.5, "output": 10.0},
    "gpt-5": {"input": 3.0, "output": 9.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "o3": {"input": 10.0, "output": 40.0},
    "o3-mini": {"input": 1.1, "output": 4.4},
    "o4-mini": {"input": 1.1, "output": 4.4},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
    "gemini-2.5-flash-lite": {"input": 0.1, "output": 0.4},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.3},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.3},
}


@dataclass
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float | None
    output_cost_usd: float | None
    total_cost_usd: float | None
    pricing_known: bool
    model: str
    provider: str


def _lookup_pricing(model: str) -> dict[str, float] | None:
    model_key = model.lower().strip()
    if model_key in PRICING_PER_MILLION:
        return PRICING_PER_MILLION[model_key]

    for key, pricing in PRICING_PER_MILLION.items():
        if model_key.startswith(key):
            return pricing
    return None


def calculate_cost(response: LLMResponse) -> CostBreakdown:
    input_tokens = response.input_tokens or 0
    output_tokens = response.output_tokens or 0
    total_tokens = response.total_tokens or (input_tokens + output_tokens)

    pricing = _lookup_pricing(response.model)
    if not pricing or (input_tokens == 0 and output_tokens == 0):
        return CostBreakdown(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost_usd=None,
            output_cost_usd=None,
            total_cost_usd=None,
            pricing_known=pricing is not None,
            model=response.model,
            provider=response.provider,
        )

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
        pricing_known=True,
        model=response.model,
        provider=response.provider,
    )


def format_cost_compact_html(response: LLMResponse, cost: CostBreakdown) -> str:
    total_cost = (
        f"${cost.total_cost_usd:.4f}"
        if cost.total_cost_usd is not None
        else "—"
    )
    temp = (
        f"{response.temperature_used:g}"
        if response.temperature_used is not None
        else "default"
    )
    return f"""
    <div class="cost-bar">
        <div class="cost-stat"><span class="cost-label">Total</span><span class="cost-value">{total_cost}</span></div>
        <div class="cost-stat"><span class="cost-label">Tokens</span><span class="cost-value">{cost.total_tokens:,}</span></div>
        <div class="cost-stat"><span class="cost-label">In / Out</span><span class="cost-value">{cost.input_tokens:,} · {cost.output_tokens:,}</span></div>
        <div class="cost-stat"><span class="cost-label">Model</span><span class="cost-value">{cost.model}</span></div>
        <div class="cost-stat"><span class="cost-label">Temp</span><span class="cost-value">{temp}</span></div>
    </div>
    """


def format_cost_report(response: LLMResponse, cost: CostBreakdown) -> str:
    lines = [
        "### Request cost",
        f"- **Provider:** {cost.provider}",
        f"- **Model:** {cost.model}",
        f"- **Input tokens:** {cost.input_tokens:,}",
        f"- **Output tokens:** {cost.output_tokens:,}",
        f"- **Total tokens:** {cost.total_tokens:,}",
    ]

    if response.temperature_used is not None:
        lines.append(f"- **Temperature:** {response.temperature_used}")
    else:
        lines.append("- **Temperature:** not used (model default)")

    if cost.total_cost_usd is not None:
        lines.extend(
            [
                f"- **Input cost:** ${cost.input_cost_usd:.6f}",
                f"- **Output cost:** ${cost.output_cost_usd:.6f}",
                f"- **Total cost:** ${cost.total_cost_usd:.6f}",
            ]
        )
    elif cost.pricing_known:
        lines.append("- **Total cost:** unavailable (missing token usage)")
    else:
        lines.append(
            "- **Total cost:** unavailable (no pricing table entry for this model)"
        )

    if response.raw_usage:
        lines.append("")
        lines.append("**Raw usage metadata**")
        for key, value in response.raw_usage.items():
            lines.append(f"- `{key}`: {value}")

    return "\n".join(lines)
