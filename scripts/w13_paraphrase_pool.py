"""Paraphrased training pool for the template-only vs paraphrased-NL ablation.

Reads data/pools/train_v1.jsonl, rewrites every `nl` with a local Ollama model, keeps prompt_ids,
golds and every other field, re-runs the pool dedup (sci.data.dedup.check) against the protected
records, and writes data/pools/train_v1_para.jsonl. Rows whose paraphrase is rejected (empty,
drops a backticked identifier / type token, or collides with a protected record) fall back to the
template NL and are flagged `paraphrased: false`, so the pool stays row-aligned with train_v1.

Resumable: raw paraphrases are appended to data/pools/train_v1_para.raw.jsonl as they arrive and
reused on restart. Refuses to run while a training process (w03_train.py) is alive.
  python scripts/w13_paraphrase_pool.py [--model gemma4:e4b] [--limit N]
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

from sci.data.dedup import check  # noqa: E402

DEFAULT_MODEL = "gemma4:e4b"
PROMPT = ("Rewrite the following programming instruction in different words. Keep every backticked token, "
          "type name (such as i32, i64, f32, f64, index, tensor<...>, memref<...>), number and function name "
          "exactly as written, keep the meaning identical, and do not add or remove requirements. "
          "Answer with the rewritten instruction only, on one line, no quotes.\n\nInstruction: {nl}\n\nRewritten:")
_TOKEN_RE = re.compile(r"`[^`]+`|\b(?:i1|i8|i16|i32|i64|f16|f32|f64|index)\b|(?:tensor|memref)<[^>]*>|\b\d+\b")


def training_alive() -> bool:
    """True if a `w03_train.py` process is running (GPU contention with the paraphraser)."""
    r = subprocess.run(["pgrep", "-f", "w03_train.py"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def clean(text: str) -> str:
    t = text.strip().splitlines()
    t = t[0].strip() if t else ""
    t = re.sub(r"^(rewritten( instruction)?|instruction)\s*:\s*", "", t, flags=re.I)
    return t.strip().strip('"').strip()


def acceptable(orig: str, para: str) -> bool:
    """Non-empty, different from the template, and every identifier / type / number token preserved."""
    if not para or para.lower() == orig.lower() or len(para) < 10:
        return False
    need = set(_TOKEN_RE.findall(orig))
    return need <= set(_TOKEN_RE.findall(para))


def load_raw(path: Path) -> dict[int, str]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[int(r["prompt_id"])] = r["para"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag (plan: Gemma-4-E4B)")
    ap.add_argument("--src", default="data/pools/train_v1.jsonl")
    ap.add_argument("--out", default="data/pools/train_v1_para.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="paraphrase only the first N rows (rest keep template NL)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--force", action="store_true", help="ignore the running-training check")
    args = ap.parse_args()
    if training_alive() and not args.force:
        print("[w13] refusing to run: a w03_train.py process is alive (GPU contention); retry when training is done", file=sys.stderr)
        sys.exit(2)
    from sci.eval.ollama import list_models, ollama_generate_raw
    try:
        models = list_models()
    except Exception as e:  # noqa: BLE001
        print(f"[w13] Ollama not reachable ({type(e).__name__}: {e}); start the daemon and pull {args.model}", file=sys.stderr)
        sys.exit(2)
    if args.model not in models and f"{args.model}:latest" not in models:
        print(f"[w13] model {args.model!r} not in Ollama ({models}); run `ollama pull {args.model}`", file=sys.stderr)
        sys.exit(2)
    src = REPO / args.src; out = REPO / args.out; raw_path = out.with_suffix(".raw.jsonl")
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    todo = rows if args.limit is None else rows[: args.limit]
    raw = load_raw(raw_path)
    t0 = time.time(); n_new = 0
    with raw_path.open("a") as fraw:
        for i, r in enumerate(todo):
            pid = int(r["prompt_id"])
            if pid in raw:
                continue
            text, meta = ollama_generate_raw(PROMPT.format(nl=r["nl"]), args.model, max_tokens=args.max_tokens,
                                             temperature=args.temperature, seed=pid)
            raw[pid] = clean(text); n_new += 1
            fraw.write(json.dumps({"prompt_id": pid, "para": raw[pid], "model": args.model, "eval_count": meta["eval_count"]}) + "\n"); fraw.flush()
            if n_new % 50 == 0:
                print(f"[w13] {i + 1}/{len(todo)} paraphrased ({(time.time() - t0) / n_new:.1f}s each)", file=sys.stderr)
    # assemble candidates, then dedup against the protected records
    cands, n_bad = [], 0
    for r in rows:
        para = raw.get(int(r["prompt_id"]), "")
        ok = acceptable(r["nl"], para)
        n_bad += (int(r["prompt_id"]) in raw) and not ok
        cands.append({**r, "nl": para if ok else r["nl"], "nl_template": r["nl"], "paraphrased": ok})
    kept, report = check([c for c in cands if c["paraphrased"]])
    kept_ids = {c["prompt_id"] for c in kept}
    final = []
    for c in cands:
        if c["paraphrased"] and c["prompt_id"] not in kept_ids:
            c = {**c, "nl": c["nl_template"], "paraphrased": False}
        final.append(c)
    out.write_text("".join(json.dumps(c) + "\n" for c in final))
    n_para = sum(c["paraphrased"] for c in final)
    summary = {"model": args.model, "n_rows": len(final), "n_paraphrased": n_para, "n_rejected_token_check": n_bad,
               "n_rejected_dedup": report["n_candidates"] - report["n_kept"], "dedup": report,
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    out.with_suffix(".report.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
