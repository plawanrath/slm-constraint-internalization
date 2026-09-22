"""Surface baseline for the scope probes: the same examples, folds, and control as scripts/w09_probes.py, but the
features are built from the token prefix visible at the probe position instead of the residual stream.

For every probe example (design `use`, `name`, `def_diff`) the feature vector is
  bag(prefix)     hashed counts (log1p) of the token ids in seq[:pos+1]           (2048 dims)
  window          hashed one-hot of the last four prefix tokens                   (4 x 512 dims)
  position        pos / len(seq)                                                  (1 dim)
For `def_diff` the feature is bag(prefix to use) - bag(prefix to def) plus sign(use - def), the surface analogue
of h[use] - h[def]. Two trivial rules are reported next to the probes: for `name`, "the name token already occurs
in the prefix"; for `def_diff`, "the definition precedes the use". A residual probe that beats the surface probe
reads something the token surface does not give; one that matches it may be reading a surface confound.

  python scripts/w28_probe_surface.py --pair HuggingFaceTB/SmolLM2-360M-Instruct:base \
      --pair HuggingFaceTB/SmolLM2-360M-Instruct:smollm2-360m-instruct_ssd_s0 --ladder results/w04_residuals/ladder.jsonl

Writes results/w09_probes/surface.json. CPU only; safe to run beside a training job.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sci.probes.dataset import build_examples  # noqa: E402
from sci.probes.probe import evaluate_probe  # noqa: E402
from w09_probes import DESIGNS, load_programs, revision_for  # noqa: E402

BAG, WIN, WINK = 2048, 512, 4


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def bag(ids: list[int]) -> np.ndarray:
    v = np.zeros(BAG, np.float32)
    if ids:
        np.add.at(v, np.asarray(ids) % BAG, 1.0)
    return np.log1p(v)


def window(ids: list[int]) -> np.ndarray:
    v = np.zeros(WINK * WIN, np.float32)
    last = ids[-WINK:]
    for i, t in enumerate(reversed(last)):
        v[i * WIN + (t % WIN)] = 1.0
    return v


def surface_features(seqs: list[list[int]], examples, design: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    ex = [e for e in examples if e.design == design]
    X, rule = [], []
    for e in ex:
        s = seqs[e.seq]
        pre = s[: e.pos + 1]
        if design == "def_diff":
            pre2 = s[: e.pos2 + 1]
            f = np.concatenate([bag(pre) - bag(pre2), [float(np.sign(e.pos - e.pos2))]])
            rule.append(int(e.pos2 < e.pos))            # definition precedes the use
        else:
            f = np.concatenate([bag(pre), window(pre), [e.pos / max(len(s), 1)]])
            tok = s[e.pos] if design == "name" else None
            rule.append(int(tok is not None and tok in s[: e.pos]))  # name token already seen (name design only)
        X.append(f)
    y = np.array([e.label for e in ex], int)
    groups = np.array([e.group for e in ex], int)
    rule = np.array(rule, int)
    rule_acc = float((rule == y).mean()) if design in ("name", "def_diff") else None
    return np.stack(X), y, groups, {"rule_acc": rule_acc, "rule_acc_flipped": (1 - rule_acc) if rule_acc is not None else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", required=True, help="MODEL:CKPT (CKPT = base | tag); tokenizer only")
    ap.add_argument("--n-programs", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=200)
    ap.add_argument("--ladder", default=None)
    ap.add_argument("--max-len", type=int, default=900)
    ap.add_argument("--l2", type=float, default=1e-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/w09_probes/surface.json")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    out_p = REPO / args.out
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for pair in args.pair:
        model, ckpt = pair.rsplit(":", 1)
        tag = f"{model.split('/')[-1]}:{ckpt}"
        t0 = time.perf_counter()
        hf = AutoTokenizer.from_pretrained(model, revision=revision_for(model))

        class _Wrap:  # the probe code expects the mlx_lm wrapper shape: `_tokenizer` is the HF tokenizer
            _tokenizer = hf

            def __getattr__(self, name):
                return getattr(hf, name)

        tok = _Wrap()
        ladder_tag = ckpt if ckpt != "base" else None
        programs = load_programs(args.n_programs, args.seed, Path(args.ladder) if args.ladder else None, ladder_tag, args.n_gen)
        seqs, examples, _ = build_examples(tok, programs, seed=args.seed, max_len=args.max_len)
        log(f"[{tag}] {len(programs)} programs -> {len(seqs)} sequences, {len(examples)} examples")
        res = {}
        for d in DESIGNS:
            X, y, groups, extra = surface_features(seqs, examples, d)
            if len(y) < 20 or len(set(y.tolist())) < 2:
                continue
            r = evaluate_probe(X, y, groups, l2=args.l2, seed=args.seed)
            r.update(extra)
            res[d] = r
            log(f"[{tag}] {d}: n={r['n']} surface acc={r['acc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                f"control={r['control_acc']:.3f} rule={extra['rule_acc']}")
        out[tag] = {"model": model, "ckpt": ckpt, "n_programs": len(programs), "n_seqs": len(seqs),
                    "features": {"bag": BAG, "window": [WINK, WIN], "position": 1}, "l2": args.l2, "designs": res,
                    "elapsed_s": round(time.perf_counter() - t0, 1)}
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({t: {d: (round(v["acc"], 3), v.get("rule_acc")) for d, v in r["designs"].items()} for t, r in out.items()}, indent=1))


if __name__ == "__main__":
    main()
