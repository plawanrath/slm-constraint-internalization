"""Interface-given control: decode arith_func_200 with the gold function signature stated in the prompt.

Interface mismatches grow after on-policy training (30 -> 47 of 143 under the full stack after SSD). If the
correctness gap between on-policy students and gold-supervised ones is a signature-convention effect, giving every
model the exact signature should close it; if it is semantic, SSD should stay at or below the untrained model and
SFT-gold should stay ahead. The prompt gains one line after the task: "Use exactly this signature: <gold header>".

  python scripts/w20_interface_given.py --model HuggingFaceTB/SmolLM2-360M-Instruct \
      --cell base --cell models/ckpt/smollm2-360m-instruct_ssd_s0/best ...

Writes results/w20_interface/ladder.jsonl (ladder schema, model tag <tag>__sig_given) and scores it with
scripts/w02_functional.py into results/w20_interface/functional.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PY = str(REPO / ".venv" / "bin" / "python")
OUT = REPO / "results" / "w20_interface"
HEADER_RE = re.compile(r"func\.func\s+@[\w$.]+\s*\([^)]*\)\s*(?:->\s*[^{]+?)?\s*(?=\{)")


def gold_header(gold: str) -> str | None:
    m = HEADER_RE.search(gold)
    return " ".join(m.group(0).split()) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--cell", action="append", required=True, help="'base' or a checkpoint dir; repeatable")
    ap.add_argument("--constraints", default="none,c1_c2_c3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    from sci.constraints.parser import is_parse_valid
    from sci.eval.generate import Generator, build_prompt
    from sci.eval.verify import verify
    from sci.train.run import load_ckpt

    OUT.mkdir(parents=True, exist_ok=True)
    ladder = OUT / "ladder.jsonl"
    rows = [json.loads(l) for l in (REPO / "data/pools/arith_func_200.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if ladder.exists():
        for l in ladder.read_text().splitlines():
            if l.strip():
                d = json.loads(l); done.add((d["model"], d["constraint"], d["prompt_id"]))
    tags = []
    for cell in args.cell:
        tag = (args.model.split("/")[-1].lower() if cell == "base" else Path(cell).parent.name) + "__sig_given"
        tags.append(tag)
        todo = [(c, r) for c in args.constraints.split(",") for r in rows if (tag, c, r["prompt_id"]) not in done]
        if not todo:
            print(f"[interface] {tag}: all cells present", file=sys.stderr); continue
        gen = Generator(args.model, revision=args.revision)
        if cell != "base":
            load_ckpt(gen, REPO / cell)
        with ladder.open("a") as f:
            for constraint in args.constraints.split(","):
                t0 = time.perf_counter(); n_vv = 0; n = 0
                for r in rows:
                    if (tag, constraint, r["prompt_id"]) in done:
                        continue
                    hdr = gold_header(r.get("gold_mlir") or "")  # mined prompts (id >= 150) carry no gold and get no signature line
                    nl = r["nl"] + (f"\nUse exactly this signature: {hdr}" if hdr else "")
                    prompt = build_prompt(gen.tokenizer, nl, r["dialect"])
                    g = gen.generate(prompt, constraint=constraint, max_tokens=600, seed=0)
                    pv = is_parse_valid(g.text); vv = bool(pv) and verify(g.text)["returncode"] == 0
                    n += 1; n_vv += vv
                    f.write(json.dumps({"model": tag, "method": "sig_given", "ckpt": cell, "train_seed": 0, "constraint": constraint,
                                        "pool": "arith_func_200", "seed": 0, "prompt_id": r["prompt_id"], "nl": r["nl"], "generated": g.text,
                                        "parse_valid": pv, "verify_valid": vv, "attempts": g.attempts, "finished": g.finished,
                                        "n_tokens": len(g.tokens), "dt": round(g.dt, 3), "signature_given": hdr}) + "\n"); f.flush()
                print(f"[interface] {tag} {constraint}: verify {n_vv}/{n} in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
        del gen
    if not args.no_score:
        subprocess.run([PY, "scripts/w02_functional.py", "--ladder", str(ladder), "--out", "results/w20_interface/functional",
                        "--models", ",".join(tags), "--constraints", args.constraints], cwd=REPO, check=False)


if __name__ == "__main__":
    main()
