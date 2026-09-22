"""New-dialect scf ladder: free / c1 / c1_c2 / c1_c2_c3 on Scf-Spec-60 with the scf stack.

Per checkpoint cell the runner decodes every prompt under the four stacks with the
scf grammars (`sci.transfer.scf.grammar`), applies the region-aware C3 validator
(`sci.transfer.scf.scope`) for the C1+C2+C3 rejection loop and the v3 flag,
verifies with `mlir-opt --verify-diagnostics` in the sci-llvm container, and
scores every verify-valid generation functionally against the gold with the
`mlir-cpu-runner` differential harness (`sci.transfer.scf.functional`). Resumable:
rows present in `results/w11_scf/ladder.jsonl` are skipped; `summary.json` is
rewritten at the end.

  python scripts/w11_scf.py --cell base135 HuggingFaceTB/SmolLM2-135M-Instruct \\
      --cell ssd135 HuggingFaceTB/SmolLM2-135M-Instruct results/w03_train/<tag>/best

`--cell TAG MODEL [CKPT_DIR]` may be repeated; `--no-functional` skips the differential.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sci.eval.verify import MLIR_OPT, verify  # noqa: E402
from sci.transfer.scf import functional as F  # noqa: E402
from sci.transfer.scf.grammar import is_parse_valid_scf  # noqa: E402
from sci.transfer.scf.ladder import CONSTRAINTS, Cell, Target, load_set, run_cell, write_summary  # noqa: E402
from sci.transfer.scf.scope import accept_or_reject  # noqa: E402
from sci.transfer.scf.task import build_prompt  # noqa: E402

OUT = REPO / "results" / "w11_scf"
SETS = {"scf_spec_60": F.BENCHMARK}


def require_container() -> None:
    try:
        subprocess.run([str(MLIR_OPT), "--version"], capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"[w11] mlir-opt wrapper failed ({e}); is the sci-llvm container running? "
                 "(cd scripts/env && docker compose up -d)")


def make_generator(model: str, revision: str | None):
    from sci.transfer.scf.generator import ScfGenerator
    return ScfGenerator(model, revision=revision)


class _Functional:
    """Gold-differential scorer with the gold gate cached per prompt."""

    def __init__(self) -> None:
        self.refs = {r["source_id"]: r for r in F.load_references()}
        self.bench = F.load_benchmark()
        self.gates: dict[str, dict] = {}

    def __call__(self, generated: str, row: dict) -> dict:
        ref = self.refs.get(row["id"])
        if ref is None:
            return {"status": "no_reference"}
        gold = F.gold_for(ref, self.bench)
        if row["id"] not in self.gates:
            self.gates[row["id"]] = F.gold_gate(gold, ref)
        res = F.differential(generated, gold, ref, gate=self.gates[row["id"]])
        return {k: v for k, v in res.items() if k != "trials"}


def revision_for(repo: str) -> str | None:
    for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    return None


def parse_cells(specs: list[list[str]], method: str) -> list[Cell]:
    cells = []
    for spec in specs:
        if len(spec) not in (2, 3):
            sys.exit(f"--cell expects TAG MODEL [CKPT_DIR], got {spec}")
        tag, model = spec[0], spec[1]
        ckpt = Path(spec[2]) if len(spec) == 3 else None
        if ckpt is not None and not (ckpt / "run_config.json").exists():
            sys.exit(f"checkpoint dir {ckpt} has no run_config.json")
        cells.append(Cell(tag=tag, model=model, revision=revision_for(model), ckpt_dir=ckpt,
                          method=method if ckpt else "none"))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", action="append", nargs="+", required=True, metavar="SPEC",
                    help="TAG MODEL [CKPT_DIR]; repeatable")
    ap.add_argument("--method", default="ssd", help="method label for checkpoint cells")
    ap.add_argument("--constraints", default=",".join(CONSTRAINTS))
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-functional", action="store_true")
    args = ap.parse_args()
    require_container()
    target = Target(dialect="scf", make_generator=make_generator, build_prompt=build_prompt,
                    parse_valid=is_parse_valid_scf, scope_ok=lambda t: accept_or_reject(t)[0], verify=verify,
                    functional=None if args.no_functional else _Functional())
    pools = {name: load_set(path) for name, path in SETS.items()}
    constraints = tuple(c for c in args.constraints.split(",") if c)
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in parse_cells(args.cell, args.method):
        run_cell(cell, target, pools, constraints, OUT / "ladder.jsonl", max_tokens=args.max_tokens,
                 seed=args.seed, limit=args.limit)
    summary = write_summary(OUT / "ladder.jsonl", OUT / "summary.json")
    for model, pools_ in summary["residuals"].items():
        for pool, res in pools_.items():
            print(f"[w11] {model}|{pool}: rates={res['rates']} functional={res['functional_rate']} "
                  f"residual_pp={res['residual_pp']}")


if __name__ == "__main__":
    main()
