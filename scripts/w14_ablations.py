"""Ablation matrix: one base SSD config per model, one named factor varied per run.

Base config (w04 campaign settings): seed 0, 2k prompts, 2 epochs, G=4, 32 prompts/step, task
reward, β=0.1, λ=1, temp 0.8; SmolLM2-360M full-FT (lr 2e-5), SmolLM2-1.7B LoRA r32 (lr 1e-4).
Ablations (`--ablations a,b,...`, default all):
  lam           λ ∈ {0, 0.5, 1}
  mask_only     λ=0, β=0 (legal-mass loss alone)
  reward_only   GRPO method (no mask at rollout time)
  lora_vs_full  1.7B full fine-tune (lr 2e-5, no LoRA); 360M is already full-FT → skipped
  pool_size     --n-train ∈ {1k, 3k, 8k}; the pool has 7,257 rows, so 8k = the whole pool
  paraphrase    train on data/pools/train_v1_para.jsonl (built by scripts/w13_paraphrase_pool.py)
  temp          rollout temperature ∈ {0.5, 0.8, 1.0}
  epochs        ∈ {1, 2, 3}
  no_c3_reward  reward kind `task_noc3` (C3 dropped from the verifier reward); skipped with a note
                until the driver exposes that kind
  c3_dense      --c3-mask (offline per-token C3 mass loss)
Runs whose settings equal the w04 baseline (`<model>_ssd_s0` / `<model>_grpo_s0`) are marked
`covered_by` in the manifest and skipped unless --force-duplicates. Each run is tagged
`<model>_<method>_s0__<ablation>=<value>`, trained with scripts/w03_train.py (skipped when its
summary.json exists) and evaluated with scripts/w04_eval_ckpt.py (skipped when the tag is already
in results/w04_residuals/residuals.json). Manifest: results/w14_ablations/manifest.json.
  python scripts/w14_ablations.py --dry-run --ablations lam,mask_only
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = ".venv/bin/python"  # relative: commands run with cwd=REPO and are stored in the manifest
OUT_DIR = "results/w14_ablations"
REPOS = {"SmolLM2-360M-Instruct": "HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-1.7B-Instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct"}
LORA = {"SmolLM2-1.7B-Instruct"}
BASE = {"method": "ssd", "seed": 0, "n_train": 2000, "epochs": 2, "group": 4, "batch_prompts": 32, "eval_every": 25,
        "reward": "task", "beta": 0.1, "lam": 1.0, "temp": 0.8}
ALL_ABLATIONS = ("lam", "mask_only", "reward_only", "lora_vs_full", "pool_size", "paraphrase", "temp", "epochs", "no_c3_reward", "c3_dense")
POOL_ROWS = 7257
PARA_POOL = "data/pools/train_v1_para.jsonl"
NOC3_KIND = "task_noc3"


def revision_for(repo: str) -> str | None:
    for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
        if line.startswith(repo + "@"):
            return line.split("@", 1)[1].strip()
    return None


def driver_has_reward_kind(kind: str) -> bool:
    """True when sci.train.spine.reward_fn knows `kind` (checked on the source, no MLX import)."""
    src = (REPO / "sci/train/spine.py").read_text()
    return kind in src


def fmt(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def variants(model: str, ablation: str) -> list[dict]:
    """(value, overrides, note) per run of one ablation for one model. `lora`/`lr` overrides replace the model default."""
    if ablation == "lam":
        return [{"value": fmt(v), "over": {"lam": v}} for v in (0.0, 0.5, 1.0)]
    if ablation == "mask_only":
        return [{"value": "1", "over": {"lam": 0.0, "beta": 0.0}}]
    if ablation == "reward_only":
        return [{"value": "1", "over": {"method": "grpo"}}]
    if ablation == "lora_vs_full":
        if model not in LORA:
            return [{"value": "full", "skip": f"{model} is already full-FT in the base config"}]
        return [{"value": "full", "over": {"lora": False, "lr": 2e-5}}]
    if ablation == "pool_size":
        out = []
        for n in (1000, 3000, 8000):
            v = {"value": f"{n // 1000}k", "over": {"n_train": n}}
            if n > POOL_ROWS:
                v["note"] = f"pool has {POOL_ROWS} rows; n_train={n} uses the whole pool"
            out.append(v)
        return out
    if ablation == "paraphrase":
        v = {"value": "para", "over": {"train_pool": PARA_POOL}}
        if not (REPO / PARA_POOL).exists():
            v["skip"] = f"{PARA_POOL} missing; build it with scripts/w13_paraphrase_pool.py"
        return [v]
    if ablation == "temp":
        return [{"value": fmt(t), "over": {"temp": t}} for t in (0.5, 0.8, 1.0)]
    if ablation == "epochs":
        return [{"value": str(e), "over": {"epochs": e}} for e in (1, 2, 3)]
    if ablation == "no_c3_reward":
        v = {"value": "1", "over": {"reward": NOC3_KIND}}
        if not driver_has_reward_kind(NOC3_KIND):
            v["skip"] = f"not yet supported: reward kind {NOC3_KIND!r} is not in sci/train/spine.py::reward_fn"
        return [v]
    if ablation == "c3_dense":
        return [{"value": "1", "over": {"c3_mask": True}}]
    raise KeyError(ablation)


def build(model: str, ablation: str, var: dict) -> dict:
    """Full config + train/eval commands for one run."""
    repo = REPOS[model]; rev = revision_for(repo)
    cfg = dict(BASE, model=repo, revision=rev, lora=model in LORA, lr=(1e-4 if model in LORA else 2e-5), tag="")
    cfg.update(var.get("over", {}))
    tag = f"{model.lower()}_{cfg['method']}_s{cfg['seed']}__{ablation}={var['value']}"
    cfg["tag"] = tag
    cmd = [PY, "scripts/w03_train.py", "--model", repo, "--method", cfg["method"], "--seed", str(cfg["seed"]),
           "--epochs", str(cfg["epochs"]), "--group", str(cfg["group"]), "--batch-prompts", str(cfg["batch_prompts"]),
           "--eval-every", str(cfg["eval_every"]), "--lr", str(cfg["lr"]), "--n-train", str(cfg["n_train"]),
           "--reward", cfg["reward"], "--beta", str(cfg["beta"]), "--lam", str(cfg["lam"]), "--temp", str(cfg["temp"]),
           "--out-dir", OUT_DIR, "--tag", tag]
    if rev:
        cmd += ["--revision", rev]
    if cfg["lora"]:
        cmd.append("--lora")
    if cfg.get("train_pool"):
        cmd += ["--train-pool", cfg["train_pool"]]
    if cfg.get("c3_mask"):
        cmd.append("--c3-mask")
    ckpt = f"models/ckpt/{tag}/best"
    eval_cmd = [PY, "scripts/w04_eval_ckpt.py", "--model", repo, "--ckpt", ckpt, "--tag", tag,
                "--method", cfg["method"], "--train-seed", str(cfg["seed"])]
    # identical to a w04 baseline run?
    baseline = dict(BASE, model=repo, revision=rev, lora=model in LORA, lr=(1e-4 if model in LORA else 2e-5), tag="")
    baseline["method"] = cfg["method"]
    same = all(cfg.get(k) == baseline.get(k) for k in set(cfg) | set(baseline) if k != "tag")
    entry = {"model": model, "ablation": ablation, "value": var["value"], "overrides": var.get("over", {}),
             "config": cfg, "train_cmd": cmd, "eval_cmd": eval_cmd, "summary": f"{OUT_DIR}/{tag}/summary.json", "ckpt": ckpt}
    if same and cfg["method"] in ("ssd", "grpo"):
        entry["covered_by"] = f"{model.lower()}_{cfg['method']}_s{cfg['seed']}"
    for k in ("note", "skip"):
        if k in var:
            entry[k] = var[k]
    return entry


def training_alive() -> bool:
    r = subprocess.run(["pgrep", "-f", "w03_train.py"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="SmolLM2-360M-Instruct,SmolLM2-1.7B-Instruct")
    ap.add_argument("--ablations", default=",".join(ALL_ABLATIONS))
    ap.add_argument("--dry-run", action="store_true", help="print the command list, write the manifest, run nothing")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--force-duplicates", action="store_true", help="also run configs identical to a w04 baseline")
    args = ap.parse_args()
    out = REPO / OUT_DIR; out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    plan = []
    for model in args.models.split(","):
        for ab in args.ablations.split(","):
            if ab not in ALL_ABLATIONS:
                raise SystemExit(f"unknown ablation {ab!r}; choose from {ALL_ABLATIONS}")
            for var in variants(model, ab):
                e = build(model, ab, var)
                manifest[e["config"]["tag"]] = {k: v for k, v in e.items() if k not in ("train_cmd", "eval_cmd")} | {"train_cmd": e["train_cmd"]}
                plan.append(e)
    mpath.write_text(json.dumps(manifest, indent=2))
    res_path = REPO / "results/w04_residuals/residuals.json"
    evaluated = set(json.loads(res_path.read_text())) if res_path.exists() else set()
    if not args.dry_run and training_alive():
        raise SystemExit("[w14] a w03_train.py process is already running; refusing to start a second one")
    for e in plan:
        tag = e["config"]["tag"]
        if "skip" in e:
            print(f"[w14] SKIP {tag}: {e['skip']}"); continue
        if "covered_by" in e and not args.force_duplicates:
            print(f"[w14] SKIP {tag}: identical to baseline {e['covered_by']} (use --force-duplicates to rerun)"); continue
        if "note" in e:
            print(f"[w14] NOTE {tag}: {e['note']}")
        done = (REPO / e["summary"]).exists()
        print(f"[w14] {'done ' if done else 'train'} {tag}\n    " + " ".join(e["train_cmd"]))
        if not done and not args.dry_run:
            subprocess.run(e["train_cmd"], check=False, cwd=REPO)
        ckpt = REPO / e["ckpt"]
        if args.no_eval:
            continue
        if tag in evaluated:
            print(f"[w14] eval  {tag}: already in residuals.json"); continue
        print(f"[w14] eval  {tag}\n    " + " ".join(e["eval_cmd"]))
        if not args.dry_run:
            if not ckpt.exists():
                print(f"[w14] eval  {tag}: no checkpoint at {ckpt}, skipping"); continue
            subprocess.run(e["eval_cmd"], check=False, cwd=REPO)
    print(f"[w14] manifest: {mpath} ({len(manifest)} tags)")


if __name__ == "__main__":
    main()
