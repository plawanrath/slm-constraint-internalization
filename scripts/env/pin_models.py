"""Resolve and download the pinned model revisions into the HF cache and write models.lock.txt.

Usage: python scripts/env/pin_models.py [--revision REPO@SHA ...]
Without --revision, the current main revision of each repo is resolved and pinned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = [
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "google/gemma-3-270m-it",
    "google/gemma-3-1b-it",
]
LOCK = Path(__file__).resolve().parent / "models.lock.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", action="append", default=[], help="REPO@SHA override")
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    overrides = dict(r.split("@", 1) for r in args.revision)
    api = HfApi()
    lines = ["# Pinned upstream model revisions (HF commit hashes). Regenerate with scripts/env/pin_models.py.",
             "# Format: <hf_repo>@<revision>"]
    for repo in MODELS:
        try:
            sha = overrides.get(repo) or api.model_info(repo).sha
            if not args.no_download:
                snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja"])
            lines.append(f"{repo}@{sha}")
            print(f"[pin] {repo}@{sha}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            lines.append(f"# {repo}  UNRESOLVED: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
            print(f"[pin] FAILED {repo}: {e}", file=sys.stderr)
    LOCK.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
