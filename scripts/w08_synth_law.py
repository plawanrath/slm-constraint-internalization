"""Synthetic scaling law: sizes × D × seeds → Residual_3(D, size) → a · D^α · size^−β.

  python scripts/w08_synth_law.py --models SmolLM2-135M-Instruct,SmolLM2-360M-Instruct --D 2,4,8,16,32,64 --seeds 0,1,2

Per (model, D, seed): (a) build a synthetic train/dev pool for that D, (b) train with the driver
(`scripts/w03_train.py --task synth ...`, skipped when its summary.json exists or with --no-train),
(c) decode the dev pool free and C1+C2-masked from the best checkpoint and record per-prompt C3
verdicts, (d) paired-bootstrap Residual_3 = v3(free) − v3(masked). Finally fit the law over the
seed-averaged points. Outputs under --out-dir: pools/, evals/{tag}.jsonl, sweep.jsonl, fit.json.
Steps (b)/(c) need the Task hook in the driver (see sci/synth/task.py); --dry-run prints the
commands and --fit-only refits from an existing sweep.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PY = str(REPO / ".venv/bin/python")

from sci.synth.generator import make_pool  # noqa: E402
from sci.synth.law import aggregate_seeds, fit_power_law, model_size, residual_from_rows  # noqa: E402
from sci.synth.task import SynthTask  # noqa: E402

REPOS = {"SmolLM2-135M-Instruct": "HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct",
         "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B"}
LORA = {"SmolLM2-1.7B-Instruct", "SmolLM3-3B"}


def revision_for(repo: str) -> str | None:
    lock = REPO / "scripts/env/models.lock.txt"
    if not lock.exists():
        return None
    for line in lock.read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    return None


def build_pools(out: Path, D: int, N: int, nesting: int, n_train: int, n_dev: int, pool_seed: int) -> tuple[Path, Path]:
    """Train/dev pools for one D (dev uses a disjoint generator seed range)."""
    pools = out / "pools"; pools.mkdir(parents=True, exist_ok=True)
    tr, dv = pools / f"train_D{D}.jsonl", pools / f"dev_D{D}.jsonl"
    if not tr.exists():
        make_pool(n_train, D, N, nesting, seed=pool_seed, path=tr)
    if not dv.exists():
        make_pool(n_dev, D, N, nesting, seed=pool_seed + 500, path=dv, id_base=10_000)
    return tr, dv


def train_cmd(model: str, seed: int, tag: str, tr: Path, dv: Path, out_dir: str, args) -> list[str]:
    repo = REPOS[model]; rev = revision_for(repo); lora = model in LORA
    cmd = [PY, "scripts/w03_train.py", "--model", repo, "--method", args.method, "--seed", str(seed),
           "--epochs", str(args.epochs), "--group", str(args.group), "--batch-prompts", str(args.batch_prompts),
           "--eval-every", str(args.eval_every), "--lr", str(args.lr or (1e-4 if lora else 2e-5)),
           "--max-tokens", str(args.max_tokens), "--train-pool", str(tr), "--dev-pool", str(dv),
           "--n-dev", str(args.n_dev), "--out-dir", out_dir, "--tag", tag, "--task", "synth"]
    if rev:
        cmd += ["--revision", rev]
    if lora:
        cmd.append("--lora")
    return cmd


def decode_dev(model: str, ckpt: Path | None, dev_rows: list[dict], max_tokens: int,
               c3_max_attempts: int = 5, retry_temp: float = 0.8, seed: int = 0) -> list[dict]:
    """Decode the dev pool under three arms and record per-prompt C3 verdicts (needs MLX):
    free (greedy), masked (greedy under the C1+C2 grammar), and c3 (C1+C2 grammar plus the scope
    validator as rejection sampling: greedy first attempt, then seeded temp-0.8 retries, up to
    `c3_max_attempts`), mirroring the main ladder's c1_c2_c3 stack. Residual_3 = v3(free) − v3(c3)."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from sci.eval.generate import Generator
    from sci.masks.llg import GrammarLogitsProcessor
    task = SynthTask()
    gen = Generator(REPOS[model], revision=revision_for(REPOS[model]))
    if ckpt is not None:
        from sci.train.run import load_ckpt
        load_ckpt(gen, ckpt)
    grammar = task.grammar(task.grammar_name)

    def run(ids, masked: bool, temp: float) -> str:
        proc = GrammarLogitsProcessor(gen.llg_tok, grammar) if masked else None
        out = []
        for tok, _ in generate_step(mx.array(ids), gen.model, max_tokens=max_tokens, sampler=make_sampler(temp=temp),
                                    logits_processors=[proc] if proc else None):
            tok = int(tok)
            if tok in gen.eos_ids:
                break
            out.append(tok)
            if proc is not None and proc.matcher.is_stopped():
                break
        return gen.tokenizer.decode(out)

    def ok(rep: dict) -> bool:
        return bool(rep["parse_valid"] and rep["scope_ok"])

    rows = []
    for r in dev_rows:
        ids = gen.encode(task.build_prompt(gen.tokenizer, r["nl"]))
        mx.random.seed(seed)
        text_free = run(ids, False, 0.0)
        mx.random.seed(seed)
        text_masked = run(ids, True, 0.0)
        rep_f, rep_m = task.check(text_free), task.check(text_masked)
        # C3 arm: the greedy masked sample is attempt 1; retries resample under the mask.
        text_c3, rep_c3, attempts = text_masked, rep_m, 1
        while not ok(rep_c3) and attempts < c3_max_attempts:
            attempts += 1
            mx.random.seed(seed * 1000 + attempts)
            text_c3 = run(ids, True, retry_temp)
            rep_c3 = task.check(text_c3)
        rows.append({"prompt_id": r["prompt_id"], "D": r["D"], "scope_ok_free": ok(rep_f), "scope_ok_masked": ok(rep_m),
                     "scope_ok_c3": ok(rep_c3), "attempts_c3": attempts, "check_free": rep_f, "check_masked": rep_m,
                     "check_c3": rep_c3, "text_free": text_free, "text_masked": text_masked, "text_c3": text_c3})
    return rows


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_fit(out: Path, sweep_rows: list[dict]) -> dict:
    points = aggregate_seeds(sweep_rows)
    fit = {"points": points, "n_points": len(points)}
    if len(points) >= 3:
        fit["fit"] = fit_power_law(points)
    (out / "fit.json").write_text(json.dumps(fit, indent=2))
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="SmolLM2-135M-Instruct,SmolLM2-360M-Instruct,SmolLM2-1.7B-Instruct")
    ap.add_argument("--D", default="2,4,8,16,32,64")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--nesting", type=int, default=1)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--pool-seed", type=int, default=0)
    ap.add_argument("--method", default="ssd")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--batch-prompts", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out-dir", default="results/w08_synth_law")
    ap.add_argument("--no-train", action="store_true", help="skip training; evaluate whatever checkpoint exists (or the base model)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build pools and print the training commands only")
    ap.add_argument("--fit-only", action="store_true", help="refit from the existing sweep.jsonl")
    ap.add_argument("--reeval", action="store_true", help="re-decode and rescore tags already in sweep.jsonl (rows are replaced)")
    args = ap.parse_args()
    out = (REPO / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sweep_path = out / "sweep.jsonl"
    if args.fit_only:
        print(json.dumps(write_fit(out, load_jsonl(sweep_path)), indent=2)); return
    done = {(r["model"], r["D"], r["seed"]) for r in load_jsonl(sweep_path)} if sweep_path.exists() and not args.reeval else set()
    Ds = [int(d) for d in args.D.split(",")]; seeds = [int(s) for s in args.seeds.split(",")]
    for D in Ds:
        tr, dv = build_pools(out, D, args.N, args.nesting, args.n_train, args.n_dev, args.pool_seed)
        for model in args.models.split(","):
            for seed in seeds:
                tag = f"synth_{model.lower()}_D{D}_s{seed}"
                if (model, D, seed) in done:
                    continue
                run_dir = out / "train" / tag
                if not args.no_train and not (run_dir / "summary.json").exists():
                    cmd = train_cmd(model, seed, tag, tr, dv, str(out / "train"), args)
                    print("[w08]", "dry-run" if args.dry_run else "train", " ".join(cmd), file=sys.stderr)
                    if not args.dry_run:
                        subprocess.run(cmd, check=False, cwd=REPO)
                if args.dry_run or args.no_eval:
                    continue
                ckpt = out / "train" / "ckpt" / tag / "best"
                if not ckpt.exists():
                    ckpt = REPO / "models" / "ckpt" / tag / "best"
                    if not ckpt.exists():
                        if not args.no_train:
                            print(f"[w08] no checkpoint for {tag}; skipping eval", file=sys.stderr); continue
                        ckpt = None
                ev_rows = decode_dev(model, ckpt, load_jsonl(dv), args.max_tokens)
                (out / "evals").mkdir(exist_ok=True)
                (out / "evals" / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ev_rows))
                res = residual_from_rows(ev_rows)
                row = {"model": model, "size": model_size(model), "D": D, "seed": seed, "tag": tag, "ckpt": str(ckpt), **res}
                prev = [x for x in load_jsonl(sweep_path) if x.get("tag") != tag] if sweep_path.exists() else []
                sweep_path.write_text("".join(json.dumps(x) + "\n" for x in prev + [row]))
                print(f"[w08] {tag} residual {res['residual']:.4f} [{res['ci_low']:.4f}, {res['ci_high']:.4f}]", file=sys.stderr)
    if sweep_path.exists():
        print(json.dumps(write_fit(out, load_jsonl(sweep_path)).get("fit"), indent=2))


if __name__ == "__main__":
    main()
