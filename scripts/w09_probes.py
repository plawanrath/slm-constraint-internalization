"""Scope probes and activation patching per (model, checkpoint).

For every `--pair MODEL:CKPT` (CKPT = "base", a checkpoint tag under models/ckpt/<tag>/best or
results/w03_train/<tag>/best, or a directory), sample `--n-programs` dev-pool golds (plus the
model's own free generations from a ladder JSONL if `--ladder` is given), build the probe
datasets of `sci.probes.dataset`, train the per-layer probes of `sci.probes.probe`, run activation
patching on `--n-patch` clean/corrupted pairs, and write results/w09_probes/{probes.json,
patching.json, summary.md}. Resumable: a (model, ckpt) already present in both JSON files is
skipped unless `--force`.

  python scripts/w09_probes.py --pair HuggingFaceTB/SmolLM2-360M-Instruct:base \\
      --pair HuggingFaceTB/SmolLM2-360M-Instruct:smollm2-360m-instruct_ssd_s0 --ladder results/w04_residuals/ladder.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from sci.constraints.parser import is_parse_valid
from sci.eval.generate import Generator
from sci.eval.ladder import POOL_DIALECT
from sci.probes.dataset import build_examples, design_matrix, extract_features
from sci.probes.patching import run_patching
from sci.probes.probe import DESIGNS, probe_by_layer

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/w09_probes"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def revision_for(repo: str) -> str | None:
    p = REPO / "scripts/env/models.lock.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    return None


def resolve_ckpt(ckpt: str) -> Path | None:
    if ckpt == "base":
        return None
    for cand in (Path(ckpt), REPO / "models/ckpt" / ckpt / "best", REPO / "results/w03_train" / ckpt / "best"):
        if (cand / "run_config.json").exists():
            return cand
    raise FileNotFoundError(f"checkpoint {ckpt!r} not found")


def load_programs(n: int, seed: int, ladder: Path | None, ladder_tag: str | None, n_gen: int) -> list[dict]:
    rows = [json.loads(l) for l in (REPO / "data/pools/dev_v1.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    progs = [{"text": r["mlir"], "nl": r["nl"], "dialect": r["dialect"], "source": "gold", "id": r["id"]}
             for r in rows[:n]]
    if ladder and ladder_tag:
        seen: set[str] = set()
        gens = []
        for line in ladder.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["model"] != ladder_tag or r["constraint"] != "none" or not r.get("parse_valid"):
                continue
            t = r["generated"].strip() + "\n"
            if t in seen or not is_parse_valid(t):
                continue
            seen.add(t)
            gens.append({"text": t, "nl": r["nl"], "dialect": POOL_DIALECT[r["pool"]], "source": "gen",
                         "id": f"{r['pool']}:{r['prompt_id']}"})
        rng.shuffle(gens)
        progs += gens[:n_gen]
        log(f"[programs] {len(gens)} parse-valid free generations for {ladder_tag}, using {min(n_gen, len(gens))}")
    return progs


def load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def write_summary(probes: dict, patching: dict, path: Path) -> None:
    lines = ["# Scope probes and activation patching", ""]
    for tag, res in probes.items():
        lines += [f"## {tag}", "", f"programs: {res['n_programs']} (gold {res['n_gold']}, gen {res['n_gen']}); "
                  f"sequences: {res['n_seqs']}; layers: {len(res['layers'])}", ""]
        lines += ["| design | n | pos rate | best layer | acc [95% CI] | control | first layer | last layer |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in DESIGNS:
            pl = res["designs"].get(d)
            if not pl:
                continue
            best = max(pl, key=lambda l: pl[l]["acc"])
            b, f, la = pl[best], pl[min(pl, key=int)], pl[max(pl, key=int)]
            lines.append(f"| {d} | {b['n']} | {b['pos_rate']:.2f} | {best} | {b['acc']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}] "
                         f"| {b['control_acc']:.3f} | {f['acc']:.3f} | {la['acc']:.3f} |")
        lines.append("")
        pt = patching.get(tag)
        if pt:
            pl = pt["per_layer"]
            best = max(pl, key=lambda l: pl[l]["target_recovery"] if pl[l]["target_recovery"] == pl[l]["target_recovery"] else -1)
            lines += [f"patching: {pt['n_pairs']} pairs ({pt['n_valid']} with a clean-corrupt gap > 0.1 nat); "
                      f"target log-prob clean {pt['target_logprob_clean']:.2f} vs corrupt {pt['target_logprob_corrupt']:.2f}; "
                      f"peak target recovery at layer {best}: {pl[best]['target_recovery']:.2f} "
                      f"[{pl[best]['target_recovery_ci'][0]:.2f}, {pl[best]['target_recovery_ci'][1]:.2f}], "
                      f"margin delta {pl[best]['margin_delta']:.2f}", ""]
            lines += ["| layer | target recovery | margin delta |", "|---|---|---|"]
            for l in sorted(pl, key=int):
                r = pl[l]
                lines.append(f"| {l} | {r['target_recovery']:.2f} [{r['target_recovery_ci'][0]:.2f}, {r['target_recovery_ci'][1]:.2f}] "
                             f"| {r['margin_delta']:.2f} [{r['margin_delta_ci'][0]:.2f}, {r['margin_delta_ci'][1]:.2f}] |")
            lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="MODEL:CKPT (CKPT = base | tag | dir)")
    ap.add_argument("--n-programs", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=200, help="free generations from --ladder to add")
    ap.add_argument("--ladder", default=None)
    ap.add_argument("--ladder-tag", default=None, help="ladder model tag for base (defaults to the ckpt tag)")
    ap.add_argument("--n-patch", type=int, default=200)
    ap.add_argument("--layers", default=None, help="comma-separated block indices (default all)")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=900)
    ap.add_argument("--l2", type=float, default=1e-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    probes_p, patch_p = out / "probes.json", out / "patching.json"
    probes, patching = load_json(probes_p), load_json(patch_p)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    for pair in args.pair:
        model, ckpt = pair.rsplit(":", 1)
        tag = f"{model.split('/')[-1]}:{ckpt}"
        if not args.force and tag in probes and tag in patching:
            log(f"[skip] {tag} done"); continue
        t0 = time.perf_counter()
        gen = Generator(model, revision=revision_for(model))
        ckpt_dir = resolve_ckpt(ckpt)
        if ckpt_dir is not None:
            from sci.train.run import load_ckpt
            load_ckpt(gen, ckpt_dir)
        ladder_tag = args.ladder_tag or (ckpt if ckpt != "base" else None)
        programs = load_programs(args.n_programs, args.seed, Path(args.ladder) if args.ladder else None,
                                 ladder_tag, args.n_gen)
        seqs, examples, pairs = build_examples(gen.tokenizer, programs, seed=args.seed, max_len=args.max_len)
        log(f"[{tag}] {len(programs)} programs -> {len(seqs)} sequences, {len(examples)} examples, {len(pairs)} pairs")
        feats, row = extract_features(gen.model, seqs, examples, layers, args.micro_batch, log)
        designs = {}
        for d in DESIGNS:
            X, y, groups = design_matrix(feats, row, examples, d)
            if len(y) < 20 or len(set(y.tolist())) < 2:
                log(f"[{tag}] design {d}: too few examples ({len(y)})"); continue
            designs[d] = probe_by_layer(X, y, groups, l2=args.l2, seed=args.seed)
            best = max(designs[d], key=lambda l: designs[d][l]["acc"])
            log(f"[{tag}] {d}: n={len(y)} best layer {best} acc={designs[d][best]['acc']:.3f} "
                f"control={designs[d][best]['control_acc']:.3f}")
        probes[tag] = {"model": model, "ckpt": ckpt, "n_programs": len(programs),
                       "n_gold": sum(p["source"] == "gold" for p in programs),
                       "n_gen": sum(p["source"] == "gen" for p in programs), "n_seqs": len(seqs),
                       "layers": sorted(int(l) for l in feats), "l2": args.l2,
                       "designs": {d: {str(l): r for l, r in res.items()} for d, res in designs.items()}}
        probes_p.write_text(json.dumps(probes, indent=1))
        del feats
        rng = random.Random(args.seed); rng.shuffle(pairs)
        sel = pairs[:args.n_patch]
        res = run_patching(gen.model, sel, layers, args.micro_batch, log)
        res["per_layer"] = {str(l): r for l, r in res["per_layer"].items()}
        patching[tag] = dict(res, model=model, ckpt=ckpt)
        patch_p.write_text(json.dumps(patching, indent=1))
        write_summary(probes, patching, out / "summary.md")
        log(f"[{tag}] done in {time.perf_counter() - t0:.0f}s")
    write_summary(probes, patching, out / "summary.md")


if __name__ == "__main__":
    main()
