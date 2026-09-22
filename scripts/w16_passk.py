"""Masked pass@k of an untrained model on the released arith+func pool: the ceiling any on-policy method can reach.

For every prompt of arith_func_200, sample k completions under the C1+C2 grammar mask at temperature 0.8
(seeded per sample index), verify each, and score every sample with the gold-differential harness
(sci.eval.differential, five randomized trials, same as the functional tables). pass@k is the fraction
of gold-gate prompts with at least one matching sample among the first k, for k = 1..K; greedy correctness under
the same mask is read from results/w02_functional for comparison.

  python scripts/w16_passk.py --model HuggingFaceTB/SmolLM2-360M-Instruct --k 16

Writes results/w16_passk/<model>/samples.jsonl, per-sample-index differential outputs, and passk.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PY = str(REPO / ".venv" / "bin" / "python")
K_TRIALS = 5  # sci.eval.differential.K_TRIALS


def _other_scorer_waiting() -> bool:
    """True when another queue is polling the scorer lock (its shell sits in a `sleep 30` between mkdir attempts)."""
    try:
        out = subprocess.run(["pgrep", "-f", "sleep 30"], capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:
        return False


def score_dev(out: Path, rows: list[dict], args) -> None:
    """In-distribution pass@k: a sample counts when it verifies and matches the template gold on one differential trial."""
    from sci.train.functional_reward import functional_match
    samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines() if l.strip()]
    golds = {i: r["mlir"] for i, r in enumerate(rows)}
    match: dict[int, list[bool]] = {}
    for s in sorted(samples, key=lambda x: (x["prompt_id"], x["sample"])):
        ok = bool(s["verify_valid"]) and functional_match(s["generated"], golds[s["prompt_id"]], trial=0)["match"]
        match.setdefault(s["prompt_id"], []).append(ok)
    n = len(rows)
    curve = [{"k": k, "n_prompts": n, "n_pass": sum(1 for pid in match if any(match[pid][:k])),
              "pass_at_k": round(sum(1 for pid in match if any(match[pid][:k])) / n, 4)} for k in range(1, args.k + 1)]
    res = {"model": args.model, "ckpt": args.ckpt, "k": args.k, "temp": args.temp, "constraint": "c1_c2", "pool": "dev_v1", "n_gate": n,
           "curve": curve, "per_prompt": {str(pid): match[pid] for pid in sorted(match)}}
    (out / "passk.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(curve))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--pool", default="data/pools/arith_func_200.jsonl")
    ap.add_argument("--out-dir", default="results/w16_passk")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="checkpoint dir to load (default: base weights)")
    ap.add_argument("--tag", default=None, help="output subdirectory (default: model name, or the checkpoint's tag)")
    ap.add_argument("--dev", action="store_true", help="in-distribution mode: first --n-dev rows of data/pools/dev_v1.jsonl, scored by one differential trial against the template gold (as scripts/w14_dev_functional.py)")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--gate-only", action="store_true", help="released pool: only the prompts with gold (prompt_id < 150), the scorable set")
    ap.add_argument("--gen-only", action="store_true", help="sample (or extend to --k) and stop; scoring runs later with --score-only")
    ap.add_argument("--score-only", action="store_true", help="score existing samples without loading the model")
    ap.add_argument("--chunk", type=int, default=16, help="sample indices generated per pass (memory bound)")
    ap.add_argument("--lock-dir", default=None, help="mkdir lock held while one sample cell is scored, so other scorers can interleave")
    args = ap.parse_args()

    from sci.eval.generate import Generator, build_prompt
    from sci.train.spine import collect_rollouts_batched

    tag = args.tag or (Path(args.ckpt).parent.name if args.ckpt else args.model.split("/")[-1].lower())
    if args.dev:
        tag += "__dev"
    out = REPO / args.out_dir / tag; out.mkdir(parents=True, exist_ok=True)
    pool = "data/pools/dev_v1.jsonl" if args.dev else args.pool
    rows = [json.loads(l) for l in (REPO / pool).read_text().splitlines() if l.strip()]
    if args.dev:
        rows = rows[: args.n_dev]
    if args.limit:
        rows = rows[: args.limit]
    if args.gate_only and not args.dev:
        rows = [r for r in rows if r.get("prompt_id", 0) < 150]
    samples_path = out / "samples.jsonl"
    have: dict[int, int] = {}
    if samples_path.exists():
        for l in samples_path.read_text().splitlines():
            if l.strip():
                d = json.loads(l); have[d["prompt_id"]] = max(have.get(d["prompt_id"], 0), d["sample"] + 1)
    start = min((have.get(r.get("prompt_id", i), 0) for i, r in enumerate(rows)), default=0)
    if start < args.k and not args.score_only:
        gen = Generator(args.model, revision=args.revision)
        if args.ckpt:
            from sci.train.run import load_ckpt
            load_ckpt(gen, REPO / args.ckpt)
        prompts = [(r.get("prompt_id", i), build_prompt(gen.tokenizer, r["nl"], r.get("dialect", "arith+func"))) for i, r in enumerate(rows)]
        while start < args.k:
            n_new = min(args.chunk, args.k - start)
            # sample indices [start, start + n_new); the seed moves with the index so extensions never repeat earlier samples
            ros = collect_rollouts_batched(gen, prompts, n_new, constraint="c1_c2", temp=args.temp, max_tokens=args.max_tokens,
                                           seed=12345 + start, with_masks=False, reward_kind="verify")
            idx: dict[int, int] = {}
            with samples_path.open("a") as f:
                for ro in ros:
                    j = idx.get(ro.prompt_id, 0); idx[ro.prompt_id] = j + 1
                    f.write(json.dumps({"prompt_id": ro.prompt_id, "sample": start + j, "generated": ro.text,
                                        "verify_valid": bool(ro.parts.get("verify", ro.reward > 0))}) + "\n")
            start += n_new
            print(f"[passk] {tag}: samples per prompt now {start}", file=sys.stderr)
        del gen
    elif start < args.k:
        print(f"[passk] {tag}: only {start} samples per prompt on disk, --score-only asked for {args.k}", file=sys.stderr); sys.exit(2)
    if args.gen_only:
        (out / "GEN_DONE").write_text(f"{args.k}\n"); return
    if args.dev:
        return score_dev(out, rows, args)
    samples = [json.loads(l) for l in samples_path.read_text().splitlines() if l.strip()]
    print(f"[passk] {len(samples)} samples for {len(rows)} prompts", file=sys.stderr)

    # score each sample index as one candidate set (the harness keys candidates by prompt_id)
    match: dict[int, list[bool]] = {}
    gate_ids: set[int] | None = None
    for j in range(args.k):
        cell = out / f"sample_{j:02d}"; cell.mkdir(exist_ok=True)
        cand = cell / "candidates.jsonl"
        if not (cell / "summary.json").exists():
            with cand.open("w") as f:
                for s in samples:
                    if s["sample"] == j and s["prompt_id"] < 150:
                        f.write(json.dumps({"model": tag, "dialect": "arith+func", "seed": 0, "prompt_id": s["prompt_id"],
                                            "generated": s["generated"], "verify_valid": s["verify_valid"]}) + "\n")
            print(f"[passk] scoring sample {j}", file=sys.stderr)
            if args.lock_dir:
                while True:
                    try:
                        os.mkdir(args.lock_dir); break
                    except FileExistsError:
                        time.sleep(15)
            try:
                subprocess.run([PY, "-m", "sci.eval.differential", "--out-dir", str(cell), "--candidates", str(cand), "--dialects", "arith"],
                               check=False, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                if args.lock_dir:
                    try:
                        os.rmdir(args.lock_dir)
                    except OSError:
                        pass
                    if _other_scorer_waiting():
                        time.sleep(45)  # let a scorer that polls the lock every 30 s take it
        gate = [json.loads(l) for l in (cell / "gold_gate.jsonl").read_text().splitlines() if l.strip()]
        if gate_ids is None:
            gate_ids = {g["idx"] for g in gate if g.get("gate") == "pass"}
        # differential.jsonl has one row per trial keyed by idx (= prompt_id); a candidate is correct when all K trials match
        trials: dict[int, list[bool]] = {}
        for l in (cell / "differential.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            trials.setdefault(d["idx"], []).append(bool(d.get("match", False)))
        for pid in gate_ids:
            t = trials.get(pid, [])
            match.setdefault(pid, []).append(len(t) == K_TRIALS and all(t))
    n = len(gate_ids or [])
    curve = []
    for k in range(1, args.k + 1):
        hit = sum(1 for pid in match if any(match[pid][:k]))
        curve.append({"k": k, "n_prompts": n, "n_pass": hit, "pass_at_k": round(hit / n, 4) if n else None})
    ever = sorted(pid for pid in match if any(match[pid]))
    res = {"model": args.model, "k": args.k, "temp": args.temp, "constraint": "c1_c2", "n_gate": n, "curve": curve,
           "prompts_ever_correct": ever, "per_prompt": {str(pid): match[pid] for pid in sorted(match)}}
    (out / "passk.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(curve))


if __name__ == "__main__":
    main()
