# Which Constraints Belong in the Weights?

Internalizing schema-derived structure (syntax, type domains, SSA scope, shape) in small language
models, with MLIR as the instrument.

- `sci/` — package: constraints, masks, training, synthetic language family, evaluation, probes, transfer targets.
- `scripts/` — one runner per experiment, writing to `results/wNN_<exp>/`.
- `results/` — committed summaries (JSON) and per-row generations (JSONL).
- `RUNBOOK.md` — environment setup and reproduction commands.

Hardware target: one Apple M4 Max, 128 GB unified memory.
