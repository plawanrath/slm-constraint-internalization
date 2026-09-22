"""Train one (model, method, seed) run. Example:
  python scripts/w03_train.py --model HuggingFaceTB/SmolLM2-135M-Instruct --method ssd --seed 0
"""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields

from sci.train.run import RunConfig, train

_TYPES = {"str": str, "int": int, "float": float, "str | None": str, "int | None": int}


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in fields(RunConfig):
        name = f"--{f.name.replace('_', '-')}"
        if f.type == "bool":
            if f.default is True:
                ap.add_argument(f"--no-{f.name.replace('_', '-')}", dest=f.name, action="store_false", default=True)
            else:
                ap.add_argument(name, action="store_true", default=f.default)
        elif f.default is MISSING:
            ap.add_argument(name, type=_TYPES[f.type], required=(f.name != "revision"), default=None)
        else:
            ap.add_argument(name, type=_TYPES[f.type], default=f.default)
    args = ap.parse_args()
    train(RunConfig(**{f.name: getattr(args, f.name) for f in fields(RunConfig)}))


if __name__ == "__main__":
    main()
