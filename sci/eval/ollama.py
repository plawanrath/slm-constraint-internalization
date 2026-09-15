"""Ollama REST wrapper (local paraphraser and large-model baselines).

Ollama is not run inside Docker: Docker on macOS has no Metal access and would
force CPU-only inference. It is the single documented host-level dependency
(see RUNBOOK.md).
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_TIMEOUT = 600.0  # 30B Q4 can be slow on first load


def ollama_generate_raw(
    prompt: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    seed: int = 0,
) -> tuple[str, dict]:
    """Single /api/generate call. Returns (text, metadata)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json=payload,
        timeout=DEFAULT_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("response", ""), {
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
        "total_duration_ns": data.get("total_duration", 0),
        "num_tokens": data.get("eval_count", 0),
    }



def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
