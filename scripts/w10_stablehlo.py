"""Held-out StableHLO ladder: free / c1 / c1_c2 / c1_c2_c3 on StableHLO-Spec-30 and Held-Out-200.

For each checkpoint cell (base model or a trained checkpoint directory) the runner
decodes every prompt under the four stacks with the existing `mlir_gen_stablehlo`
grammar, verifies with `iree-compile` (`sci.eval.verify_stablehlo`), and records
per-layer violation flags for the portability residuals. Resumable: rows present
in `results/w10_stablehlo/ladder.jsonl` are skipped; `summary.json` is rewritten
at the end.

  python scripts/w10_stablehlo.py --cell base135 HuggingFaceTB/SmolLM2-135M-Instruct \\
      --cell ssd135 HuggingFaceTB/SmolLM2-135M-Instruct results/w03_train/<tag>/best

`--cell TAG MODEL [CKPT_DIR]` may be repeated. `iree-compile` must be present in the
project venv (`pip install iree-compiler==<pinned>` from scripts/env/requirements.txt);
the run aborts before loading any model if it is missing.

Notes: `mlir.lark` has no StableHLO rules, so parse-valid here means "accepted by the
`mlir_gen_stablehlo` grammar" (the target's C1 definition). The scope layer is the
flat validator of `sci.constraints.scope`, which understands the ten StableHLO ops.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sci.constraints.parser import generation_grammar_path  # noqa: E402
from sci.constraints.scope import accept_or_reject  # noqa: E402
from sci.eval.verify_stablehlo import IREE_COMPILE, verify_stablehlo  # noqa: E402
from sci.transfer.scf.ladder import CONSTRAINTS, Cell, Target, load_set, run_cell, write_summary  # noqa: E402

OUT = REPO / "results" / "w10_stablehlo"
SETS = {"stablehlo_spec_30": REPO / "data/benchmarks/stablehlo_spec_30.jsonl",
        "stablehlo_held_out_200": REPO / "data/benchmarks/stablehlo_held_out_200.jsonl"}

FEW_SHOT_STABLEHLO = """Example 1:
Task: Write a function that adds two 1-D f32 tensors of 8 elements using stablehlo.add.
MLIR:
module {
  func.func @f(%a : tensor<8xf32> , %b : tensor<8xf32>) -> tensor<8xf32> {
    %0 = stablehlo.add %a , %b : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

Example 2:
Task: Write a function that transposes a 4x8 f32 tensor into an 8x4 tensor.
MLIR:
module {
  func.func @f(%a : tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %a , dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}

Example 3:
Task: Write a function that multiplies a 4x8 f32 matrix by an 8x2 f32 matrix with stablehlo.dot_general.
MLIR:
module {
  func.func @f(%a : tensor<4x8xf32> , %b : tensor<8x2xf32>) -> tensor<4x2xf32> {
    %0 = stablehlo.dot_general %a , %b , contracting_dims = [1] x [0] : (tensor<4x8xf32>, tensor<8x2xf32>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }
}
"""


def require_iree() -> None:
    if not IREE_COMPILE.exists():
        sys.exit(f"[w10] iree-compile not found at {IREE_COMPILE}.\n"
                 "      Install the pinned wheel into the project venv:\n"
                 "        .venv/bin/pip install $(grep '^iree-compiler==' scripts/env/requirements.txt)\n"
                 "      then rerun. (It is a host-side tool; the sci-llvm container does not ship StableHLO.)")
    try:
        subprocess.run([str(IREE_COMPILE), "--version"], capture_output=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        sys.exit(f"[w10] iree-compile is present but not runnable: {e}")


@lru_cache(maxsize=1)
def _stablehlo_parser():
    from lark import Lark
    return Lark(generation_grammar_path("mlir_gen_stablehlo").read_text(), parser="earley", lexer="dynamic_complete")


def parse_valid_stablehlo(text: str) -> bool:
    from lark.exceptions import LarkError
    try:
        _stablehlo_parser().parse(text)
        return True
    except LarkError:
        return False


def make_generator(model: str, revision: str | None):
    import sci.eval.generate as G
    from sci.masks.llg import lark_grammar

    G.FEW_SHOT.setdefault("stablehlo", FEW_SHOT_STABLEHLO)

    class StablehloGenerator(G.Generator):
        """Every constraint stack uses the single StableHLO generation grammar."""

        def grammar(self, name: str) -> str:
            if "mlir_gen_stablehlo" not in self._grammars:
                self._grammars["mlir_gen_stablehlo"] = lark_grammar("mlir_gen_stablehlo")
            return self._grammars["mlir_gen_stablehlo"]

    return StablehloGenerator(model, revision=revision)


def build_prompt(tokenizer, nl: str) -> str:
    import sci.eval.generate as G
    G.FEW_SHOT.setdefault("stablehlo", FEW_SHOT_STABLEHLO)
    return G.build_prompt(tokenizer, nl, "stablehlo")


TARGET = Target(dialect="stablehlo", make_generator=make_generator, build_prompt=build_prompt,
                parse_valid=parse_valid_stablehlo, scope_ok=lambda t: accept_or_reject(t)[0],
                verify=verify_stablehlo)


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
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--constraints", default=",".join(CONSTRAINTS))
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    require_iree()
    pools = {name: load_set(SETS[name]) for name in args.sets.split(",") if name}
    constraints = tuple(c for c in args.constraints.split(",") if c)
    OUT.mkdir(parents=True, exist_ok=True)
    for cell in parse_cells(args.cell, args.method):
        run_cell(cell, TARGET, pools, constraints, OUT / "ladder.jsonl", max_tokens=args.max_tokens,
                 seed=args.seed, limit=args.limit)
    summary = write_summary(OUT / "ladder.jsonl", OUT / "summary.json")
    for model, pools_ in summary["residuals"].items():
        for pool, res in pools_.items():
            print(f"[w10] {model}|{pool}: verify={res['rates']} residual_pp={res['residual_pp']}")


if __name__ == "__main__":
    main()
