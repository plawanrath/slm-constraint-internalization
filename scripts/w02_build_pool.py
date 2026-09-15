"""Build the training pool v1: template candidates → parse + generation-grammar + mlir-opt gates →
disjointness filter → train/dev split. Writes data/pools/train_v1.jsonl, data/pools/dev_v1.jsonl,
results/w02_pool/{contamination.json,families.json}.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.utils import load_tokenizer

from sci.constraints.parser import is_parse_valid
from sci.data.dedup import check, load_protected
from sci.data.templates import generate
from sci.eval.verify import verify
from sci.masks.llg import lark_grammar, llg_tokenizer, replay_allowed_sets

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-arith", type=int, default=6000)
    ap.add_argument("--n-linalg", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oversample", type=float, default=3.0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    t0 = time.time()
    cands = generate(int(args.n_arith * args.oversample), int(args.n_linalg * args.oversample), seed=args.seed)
    print(f"[pool] {len(cands)} template candidates", file=sys.stderr)
    kept, report = check(cands, load_protected())
    print(f"[pool] disjointness: {report}", file=sys.stderr)
    hf = load_tokenizer(Path(snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf); gram = lark_grammar("mlir_gen_c1c2")
    gates = collections.Counter(); good = []
    for i, r in enumerate(kept):
        if not is_parse_valid(r["mlir"]):
            gates["parse_fail"] += 1; continue
        try:
            replay_allowed_sets(tok, gram, hf.encode(r["mlir"], add_special_tokens=False))
        except ValueError:
            gates["gen_grammar_reject"] += 1; continue
        if verify(r["mlir"])["returncode"] != 0:
            gates["verify_fail"] += 1; continue
        gates["pass"] += 1; good.append(r)
        if (i + 1) % 1000 == 0:
            print(f"[pool] gated {i+1}/{len(kept)} {dict(gates)} {(time.time()-t0)/60:.1f} min", file=sys.stderr)
    report["gates"] = dict(gates)
    report["gate_pass_rate"] = round(gates["pass"] / max(1, sum(gates.values())), 4)
    rng = random.Random(args.seed)
    by_d = {"arith+func": [r for r in good if r["dialect"] == "arith+func"], "linalg": [r for r in good if r["dialect"] == "linalg"]}
    for d in by_d:
        rng.shuffle(by_d[d])
    n_dev_a = int(args.n_dev * args.n_arith / (args.n_arith + args.n_linalg)); n_dev_l = args.n_dev - n_dev_a
    dev = by_d["arith+func"][:n_dev_a] + by_d["linalg"][:n_dev_l]
    train = by_d["arith+func"][n_dev_a:n_dev_a + args.n_arith] + by_d["linalg"][n_dev_l:n_dev_l + args.n_linalg]
    for split, rows in (("train", train), ("dev", dev)):
        rng.shuffle(rows)  # mix dialects so prefixes of the file are stratified
        for i, r in enumerate(rows):
            r["prompt_id"] = i; r["split"] = split
        out = REPO / f"data/pools/{split}_{args.tag}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[pool] wrote {out} ({len(rows)} rows)", file=sys.stderr)
    report["train"] = {"n": len(train), "arith+func": sum(r["dialect"] == "arith+func" for r in train),
                       "linalg": sum(r["dialect"] == "linalg" for r in train)}
    report["dev"] = {"n": len(dev)}
    report["available_after_gates"] = {d: len(v) for d, v in by_d.items()}
    report["elapsed_min"] = round((time.time() - t0) / 60, 1)
    od = REPO / "results/w02_pool"; od.mkdir(parents=True, exist_ok=True)
    (od / "contamination.json").write_text(json.dumps(report, indent=2))
    fam = collections.Counter(r["family"] for r in train)
    (od / "families.json").write_text(json.dumps(dict(fam.most_common()), indent=2))
    print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
