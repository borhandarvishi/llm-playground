"""Prompt loading and {{variable}} substitution helpers."""

from __future__ import annotations

import re
from pathlib import Path

VARIABLE_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def extract_variables(text: str) -> list[str]:
    """Return unique variable names in first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for match in VARIABLE_PATTERN.finditer(text or ""):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def substitute_variables(text: str, values: dict[str, str]) -> str:
    result = text
    for name, value in values.items():
        result = result.replace(f"{{{{{name}}}}}", value)
    return result


def read_prompt_file(file_path: str | None) -> str:
    if not file_path:
        return ""

    path = Path(file_path)
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8")

    # If user uploads a Python prompt module, try to pull the main string assignment.
    if path.suffix == ".py":
        match = re.search(
            r'^[ \t]*(\w+)\s*=\s*"""(.*?)"""',
            content,
            flags=re.DOTALL | re.MULTILINE,
        )
        if match:
            return match.group(2).strip()
        match = re.search(
            r"^[ \t]*(\w+)\s*=\s*'''(.*?)'''",
            content,
            flags=re.DOTALL | re.MULTILINE,
        )
        if match:
            return match.group(2).strip()

    return content


def combine_prompt(prompt_text: str, file_path: str | None) -> str:
    file_content = read_prompt_file(file_path)
    if file_content and prompt_text.strip():
        return f"{file_content}\n\n{prompt_text.strip()}"
    if file_content:
        return file_content
    return prompt_text.strip()


def get_file_path(prompt_file) -> str | None:
    if not prompt_file:
        return None
    if isinstance(prompt_file, str):
        return prompt_file
    return getattr(prompt_file, "name", None)


def variables_table_from_prompt(prompt_text: str, file_path: str | None) -> list[list[str]]:
    prompt = combine_prompt(prompt_text, file_path)
    return [[name, ""] for name in extract_variables(prompt)]


def values_from_table(table) -> dict[str, str]:
    if table is None:
        return {}

    rows = table.values.tolist() if hasattr(table, "values") else table
    values: dict[str, str] = {}
    for row in rows:
        if row is None:
            continue
        if not row:
            continue
        name = str(row[0]).strip()
        if not name:
            continue
        value = str(row[1]) if len(row) > 1 and row[1] is not None else ""
        values[name] = value
    return values
