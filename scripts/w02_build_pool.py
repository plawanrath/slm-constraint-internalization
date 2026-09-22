"""Build a training pool: template candidates → disjointness filter → parse + generation-grammar +
mlir-opt gates → train/dev split. Default (v1) writes data/pools/{train,dev}_v1.jsonl and
results/w02_pool/{contamination.json,families.json}; `--families v1c --tag v1c` writes the coverage pool
data/pools/{train,dev}_v1c.jsonl and results/w02_pool/contamination_v1c.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm.utils import load_tokenizer

from sci.constraints.parser import is_parse_valid
from sci.data.dedup import check, load_protected
from sci.data.templates import V1C_FAMILIES, Gen, generate
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
    ap.add_argument("--tag", default="v1", help="output suffix: data/pools/{train,dev}_{tag}.jsonl")
    ap.add_argument("--families", default="v1", choices=list(Gen.FAMILY_SETS),
                    help="template family set: v1 (original) or v1c (v1 plus the coverage families)")
    ap.add_argument("--workers", type=int, default=4, help="threads for the mlir-opt gate (capped at 6; the container is shared)")
    args = ap.parse_args()
    t0 = time.time()
    cands = generate(int(args.n_arith * args.oversample), int(args.n_linalg * args.oversample), seed=args.seed, families=args.families)
    print(f"[pool] {len(cands)} template candidates ({args.families})", file=sys.stderr)
    kept, report = check(cands, load_protected())
    print(f"[pool] disjointness: {report}", file=sys.stderr)
    hf = load_tokenizer(Path(snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf); gram = lark_grammar("mlir_gen_c1c2")
    gates = collections.Counter(); per_family = collections.defaultdict(collections.Counter); to_verify = []
    for i, r in enumerate(kept):
        per_family[r["family"]]["n"] += 1
        if not is_parse_valid(r["mlir"]):
            gates["parse_fail"] += 1; per_family[r["family"]]["parse_fail"] += 1; continue
        try:
            replay_allowed_sets(tok, gram, hf.encode(r["mlir"], add_special_tokens=False))
        except ValueError:
            gates["gen_grammar_reject"] += 1; per_family[r["family"]]["gen_grammar_reject"] += 1; continue
        to_verify.append(r)
        if (i + 1) % 2000 == 0:
            print(f"[pool] parse+grammar {i+1}/{len(kept)} {dict(gates)} {(time.time()-t0)/60:.1f} min", file=sys.stderr)
    good = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as ex:  # verify() opens its own cache connection per call
        for i, (r, res) in enumerate(zip(to_verify, ex.map(lambda r: verify(r["mlir"]), to_verify))):
            if res["returncode"] != 0:
                gates["verify_fail"] += 1; per_family[r["family"]]["verify_fail"] += 1; continue
            gates["pass"] += 1; per_family[r["family"]]["pass"] += 1; good.append(r)
            if (i + 1) % 2000 == 0:
                print(f"[pool] verified {i+1}/{len(to_verify)} {dict(gates)} {(time.time()-t0)/60:.1f} min", file=sys.stderr)
    report["gates"] = dict(gates)
    report["gates_per_family"] = {f: {**dict(c), "pass_rate": round(c["pass"] / max(1, c["n"]), 4)} for f, c in sorted(per_family.items())}
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
    fam = collections.Counter(r["family"] for r in train)
    report["families_set"] = args.families
    report["train_families"] = dict(fam.most_common())
    report["dev_families"] = dict(collections.Counter(r["family"] for r in dev).most_common())
    n_new = sum(v for f, v in fam.items() if f in V1C_FAMILIES)
    report["train_new_family_share"] = round(n_new / max(1, len(train)), 4)
    suffix = "" if args.tag == "v1" else f"_{args.tag}"
    (od / f"contamination{suffix}.json").write_text(json.dumps(report, indent=2))
    if args.tag == "v1":
        (od / "families.json").write_text(json.dumps(dict(fam.most_common()), indent=2))
    print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
