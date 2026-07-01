"""Unified LLM client for OpenAI and Gemini chat completions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


# Reasoning models that use a fixed default temperature.
NO_TEMPERATURE_PREFIXES = ("o1", "o3", "o4")


OPENAI_MODELS = [
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o3-mini",
    "o4-mini",
]

GEMINI_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_usage: dict[str, Any]
    temperature_used: float | None
    request_error: str | None = None


def get_api_key(provider: str) -> str:
    provider = provider.lower()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "").strip()
    if provider == "gemini":
        return (
            os.getenv("GEMINI_API_KEY", "").strip()
            or os.getenv("GEMENI_API_KEY", "").strip()
        )
    raise ValueError(f"Unsupported provider: {provider}")


def model_supports_temperature(model: str) -> bool:
    model_lower = model.lower()
    return not any(model_lower.startswith(prefix) for prefix in NO_TEMPERATURE_PREFIXES)


def _openai_client() -> OpenAI:
    api_key = get_api_key("openai")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in .env")
    return OpenAI(api_key=api_key)


def _gemini_client():
    if genai is None:
        raise ValueError("google-genai is not installed. Run: pip install google-genai")
    api_key = get_api_key("gemini")
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or GEMENI_API_KEY) is not set in .env")
    return genai.Client(api_key=api_key)


def call_llm(
    *,
    provider: str,
    model: str,
    prompt: str,
    temperature: float | None,
) -> LLMResponse:
    provider = provider.lower().strip()
    model = model.strip()
    use_temperature = temperature if model_supports_temperature(model) else None

    if provider == "openai":
        return _call_openai(model=model, prompt=prompt, temperature=use_temperature)
    if provider == "gemini":
        return _call_gemini(model=model, prompt=prompt, temperature=use_temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def _call_openai(
    *,
    model: str,
    prompt: str,
    temperature: float | None,
) -> LLMResponse:
    client = _openai_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        completion = client.chat.completions.create(**kwargs)
    except Exception as exc:
        return LLMResponse(
            text="",
            provider="openai",
            model=model,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            raw_usage={},
            temperature_used=temperature,
            request_error=str(exc),
        )

    usage = completion.usage
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    text = ""
    if completion.choices:
        text = completion.choices[0].message.content or ""

    raw_usage: dict[str, Any] = {}
    if usage:
        raw_usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if hasattr(usage, "model_dump"):
            raw_usage.update(usage.model_dump())

    return LLMResponse(
        text=text,
        provider="openai",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
        temperature_used=temperature,
    )


def _call_gemini(
    *,
    model: str,
    prompt: str,
    temperature: float | None,
) -> LLMResponse:
    client = _gemini_client()
    config_kwargs: dict[str, Any] = {}
    if temperature is not None:
        config_kwargs["temperature"] = temperature

    config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        return LLMResponse(
            text="",
            provider="gemini",
            model=model,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            raw_usage={},
            temperature_used=temperature,
            request_error=str(exc),
        )

    text = getattr(response, "text", None) or ""
    usage = getattr(response, "usage_metadata", None)

    input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    total_tokens = getattr(usage, "total_token_count", None) if usage else None

    raw_usage: dict[str, Any] = {}
    if usage:
        raw_usage = {
            "prompt_token_count": input_tokens,
            "candidates_token_count": output_tokens,
            "total_token_count": total_tokens,
        }

    return LLMResponse(
        text=text,
        provider="gemini",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
        temperature_used=temperature,
    )
