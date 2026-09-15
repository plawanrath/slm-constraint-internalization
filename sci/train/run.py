"""Training driver for SSD / RFT / GRPO / SFT-gold with dev-pool evaluation and checkpoints.

A run = (model, method, seed). Each epoch iterates the training pool in prompt batches; for each
batch it samples G rollouts per prompt (SSD/RFT under the C1+C2 mask, GRPO free), replays masks,
scores rewards, and takes one optimizer step. SFT-gold uses the pool golds instead of rollouts.
Every `eval_every` steps the dev pool is decoded free and masked (C1+C2) and verify rates are
logged; the best checkpoint by dev free verify-rate is kept.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from sci.constraints.parser import is_parse_valid
from sci.eval.generate import Generator, build_prompt
from sci.eval.verify import verify
from sci.train.spine import Rollout, collect_rollouts_batched, make_lora, reward_fn, ssd_step, teacher_logprobs

REPO = Path(__file__).resolve().parents[2]


@dataclass
class RunConfig:
    model: str
    revision: str | None
    method: str            # ssd | rft | grpo | sft
    seed: int = 0
    lora: bool = False
    lora_rank: int = 32
    lr: float = 2e-5
    epochs: int = 3
    batch_prompts: int = 8
    group: int = 4
    temp: float = 0.8
    max_tokens: int = 300
    lam: float = 1.0
    train_pool: str = "data/pools/train_v1.jsonl"
    dev_pool: str = "data/pools/dev_v1.jsonl"
    n_train: int | None = None
    n_dev: int = 100
    eval_every: int = 50
    out_dir: str = "results/w03_train"
    tag: str = ""
    micro_batch: int = 4
    reward: str = "task"   # task | verify
    beta: float = 0.1      # weight of the per-token teacher term (log-ratio clipped to ±5) (fixed masked base for ssd, unmasked base for grpo)


def dev_eval(gen: Generator, dev_rows: list[dict], max_tokens: int, reward_kind: str = "task") -> dict:
    """Free and masked decoding on the dev pool: verify rate, task reward, distinct-program count
    (collapse detector), mean generated length, and unfinished rate."""
    import re as _re
    res = {}
    for constraint in ("none", "c1_c2"):
        vv, rw, progs, lens, unfin = [], [], set(), [], 0
        for r in dev_rows:
            g = gen.generate(build_prompt(gen.tokenizer, r["nl"], r["dialect"]), constraint=constraint, max_tokens=max_tokens)
            rr, parts = reward_fn(g.text, r["mlir"], reward_kind)
            vv.append(parts["verify"]); rw.append(rr); lens.append(len(g.tokens)); unfin += (not g.finished)
            progs.add(_re.sub(r"\s+", " ", g.text.strip()))
        res[constraint] = {"verify": round(sum(vv) / len(vv), 4), "reward": round(sum(rw) / len(rw), 4),
                           "distinct": len(progs), "mean_len": round(sum(lens) / len(lens), 1), "unfinished": unfin}
    return res


def save_ckpt(gen: Generator, cfg: RunConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    params = dict(tree_flatten(gen.model.trainable_parameters() if cfg.lora else gen.model.parameters()))
    mx.save_safetensors(str(path / ("adapters.safetensors" if cfg.lora else "weights.safetensors")), params)
    (path / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))


def train(cfg: RunConfig) -> dict:
    t0 = time.time()
    tag = cfg.tag or f"{cfg.model.split('/')[-1].lower()}_{cfg.method}_s{cfg.seed}"
    out = REPO / cfg.out_dir / tag; out.mkdir(parents=True, exist_ok=True)
    ckpt_root = (REPO / "models" / "ckpt" / tag) if not Path(cfg.out_dir).is_absolute() else Path(cfg.out_dir) / "ckpt" / tag
    log = (out / "train_log.jsonl").open("a")
    gen = Generator(cfg.model, revision=cfg.revision)
    if cfg.lora:
        make_lora(gen.model, rank=cfg.lora_rank)
    V = gen.llg_tok.vocab_size
    ref = Generator(cfg.model, revision=cfg.revision).model if (cfg.beta > 0 and cfg.method in ("ssd", "grpo")) else None
    if ref is not None:
        ref.freeze()
    train_rows = [json.loads(l) for l in (REPO / cfg.train_pool).read_text().splitlines() if l.strip()]
    dev_rows = [json.loads(l) for l in (REPO / cfg.dev_pool).read_text().splitlines() if l.strip()][: cfg.n_dev]
    if cfg.n_train:
        train_rows = train_rows[: cfg.n_train]
    prompts = [(r["prompt_id"], build_prompt(gen.tokenizer, r["nl"], r["dialect"]), r["mlir"]) for r in train_rows]
    golds = {r["prompt_id"]: r["mlir"] for r in train_rows}
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=0.0)
    rng = random.Random(cfg.seed)
    step = 0; best = {"dev_none": -1.0, "step": -1}
    ev = dev_eval(gen, dev_rows, cfg.max_tokens, cfg.reward)
    log.write(json.dumps({"step": 0, "event": "eval", "dev": ev, "t": round(time.time() - t0)}) + "\n"); log.flush()
    print(f"[{tag}] step 0 dev {ev}", file=sys.stderr)
    best = {"dev_none": ev["none"]["reward"], "step": 0}
    for epoch in range(cfg.epochs):
        order = list(range(len(prompts))); rng.shuffle(order)
        for s in range(0, len(order), cfg.batch_prompts):
            batch = [prompts[i] for i in order[s: s + cfg.batch_prompts]]
            if cfg.method == "sft":
                ros = []
                for pid, p, _g in batch:
                    ids = gen.encode(p); gid = gen.tokenizer.encode(golds[pid], add_special_tokens=False) + [gen.tokenizer.eos_token_id]
                    ros.append(Rollout(pid, ids, gid, golds[pid], None, 1.0))
                st = ssd_step(gen.model, opt, ros, V, mode="sft", micro_batch=cfg.micro_batch)
            else:
                constraint = "none" if cfg.method == "grpo" else "c1_c2"
                ros = collect_rollouts_batched(gen, batch, cfg.group, constraint=constraint, temp=cfg.temp,
                                               max_tokens=cfg.max_tokens, seed=cfg.seed * 7919 + step,
                                               with_masks=(cfg.method == "ssd"), reward_kind=cfg.reward)
                if ref is not None:
                    for ro in ros:
                        ro.teacher_lp = teacher_logprobs(ref, ro, V, masked=(cfg.method == "ssd"))
                st = ssd_step(gen.model, opt, ros, V, lam=cfg.lam, mode=cfg.method, micro_batch=cfg.micro_batch, beta=cfg.beta)
            step += 1
            rec = {"step": step, "epoch": epoch, "loss": st.loss, "loss_mask": st.loss_mask, "loss_grpo": st.loss_grpo,
                   "n_rollouts": st.n_rollouts, "mean_reward": st.mean_reward, "legal_mass": st.mean_legal_mass,
                   "dt": round(st.dt, 1), "t": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if not math.isfinite(st.loss):
                print(f"[{tag}] NaN/inf loss at step {step}; stopping", file=sys.stderr); break
            if step % 10 == 0:
                print(f"[{tag}] step {step} loss {st.loss:.4f} mask {st.loss_mask:.4f} reward {st.mean_reward:.2f} "
                      f"legal_mass {st.mean_legal_mass:.3f} {rec['t']}s", file=sys.stderr)
            if step % cfg.eval_every == 0:
                ev = dev_eval(gen, dev_rows, cfg.max_tokens, cfg.reward)
                log.write(json.dumps({"step": step, "event": "eval", "dev": ev, "t": round(time.time() - t0)}) + "\n"); log.flush()
                print(f"[{tag}] step {step} dev {ev}", file=sys.stderr)
                if ev["none"]["reward"] > best["dev_none"]:
                    best = {"dev_none": ev["none"]["reward"], "step": step}; save_ckpt(gen, cfg, ckpt_root / "best")
    ev = dev_eval(gen, dev_rows, cfg.max_tokens, cfg.reward)
    log.write(json.dumps({"step": step, "event": "eval_final", "dev": ev, "t": round(time.time() - t0)}) + "\n"); log.flush()
    if ev["none"]["reward"] >= best["dev_none"]:
        best = {"dev_none": ev["none"]["reward"], "step": step}; save_ckpt(gen, cfg, ckpt_root / "best")
    save_ckpt(gen, cfg, ckpt_root / "final")
    summary = {"tag": tag, "config": asdict(cfg), "steps": step, "best": best, "final_dev": ev, "ckpt_dir": str(ckpt_root),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{tag}] done: best dev free {best} final {ev} in {summary['elapsed_min']} min", file=sys.stderr)
    return summary


def load_ckpt(gen: Generator, ckpt_dir: Path) -> None:
    cfg = json.loads((ckpt_dir / "run_config.json").read_text())
    if cfg["lora"]:
        make_lora(gen.model, rank=cfg["lora_rank"])
        gen.model.load_weights(str(ckpt_dir / "adapters.safetensors"), strict=False)
    else:
        gen.model.load_weights(str(ckpt_dir / "weights.safetensors"))
    mx.eval(gen.model.parameters())
