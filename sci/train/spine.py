"""Shared training spine: rollouts → offline mask replay → rewards → losses → step.

Losses (see ADR-0003):
  legal_mass_loss : −mean_t log Σ_{v∈A_t} π_θ(v | x_<t) on masked rollouts (dense, C1/C2)
  grpo_loss       : −mean_t A(x) · log π_θ(x_t | x_<t) with group-normalized advantage from the
                    sequence reward r(x) ∈ {0,1} (parse ∧ scope ∧ mlir-opt), on-policy single step
  sft_loss        : −mean_t log π_θ(x_t | x_<t) on given target sequences (RFT / SFT-gold)
All losses take (model, prompt_ids, gen_ids) and score only generated positions.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sci.constraints.parser import is_parse_valid
from sci.constraints.scope import accept_or_reject
from sci.eval.generate import Generator
from sci.eval.verify import verify
from sci.masks.llg import bitmask_to_bool, replay_allowed_sets


@dataclass
class Rollout:
    prompt_id: int
    prompt_ids: list[int]
    gen_ids: list[int]
    text: str
    bitmask: np.ndarray | None  # (T, W) int32 allowed sets before each generated token
    reward: float
    parts: dict = field(default_factory=dict)
    teacher_lp: np.ndarray | None = None  # log π_teacher(x_t | x_<t) per generated token (fixed teacher)


_SIG_RE = re.compile(r"func\.func\s+@[\w$.]+\s*\(([^)]*)\)\s*(?:->\s*([^{]+?))?\s*\{", re.S)
_OP_RE = re.compile(r"\b((?:arith|memref|linalg|func|stablehlo|scf|math|cf|tensor)\.[a-z_]+)\b")


def signature(text: str) -> tuple[tuple[str, ...], str] | None:
    """(argument types, result type string) of the first function, whitespace-normalized."""
    m = _SIG_RE.search(text)
    if not m:
        return None
    args = tuple(re.sub(r"\s+", "", a.split(":", 1)[1]) for a in m.group(1).split(",") if ":" in a)
    ret = re.sub(r"\s+", "", m.group(2) or "")
    return args, ret


def signature_score(gen_text: str, gold_text: str) -> float:
    """Graded signature agreement in [0, 1]: per-position argument type matches plus result match,
    divided by max(#args) + 1. Exact types (memref shapes included) are required per position."""
    sg, sd = signature(gen_text), signature(gold_text)
    if sg is None or sd is None:
        return 0.0
    n = max(len(sg[0]), len(sd[0])) + 1
    hits = sum(1 for a, b in zip(sg[0], sd[0]) if a == b) + (1 if sg[1] == sd[1] else 0)
    return hits / n


def op_multiset(text: str) -> Counter:
    """Multiset of op mnemonics, excluding the function wrapper itself."""
    return Counter(o for o in _OP_RE.findall(text) if o not in ("func.func", "func.return"))


def op_jaccard(a: Counter, b: Counter) -> float:
    inter = sum((a & b).values()); union = sum((a | b).values())
    return inter / union if union else 1.0


def reward_fn(text: str, gold: str | None = None, kind: str = "task") -> tuple[float, dict]:
    """kind='verify': r = verify-valid (prompt-independent; collapses to a trivial program).
    kind='task':   r = verify · signature_score(gen, gold) · Jaccard(op multiset, gold op multiset)."""
    pv = is_parse_valid(text)
    sc = bool(pv) and accept_or_reject(text)[0]
    vv = bool(sc) and verify(text)["returncode"] == 0
    parts = {"parse": pv, "scope": sc, "verify": vv}
    if kind == "verify" or gold is None:
        return float(vv), parts
    sig = signature_score(text, gold) if vv else 0.0
    jac = op_jaccard(op_multiset(text), op_multiset(gold)) if vv else 0.0
    parts.update({"sig": round(sig, 3), "op_jaccard": round(jac, 3)})
    return float(vv) * sig * jac, parts


def rewards_parallel(texts: list[str], golds: list[str | None], kind: str = "task", workers: int = 6) -> list[tuple[float, dict]]:
    """Rewards in parallel (mlir-opt runs as a subprocess, so threads suffice)."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda tg: reward_fn(tg[0], tg[1], kind), zip(texts, golds)))


def collect_rollouts(gen: Generator, prompts: list[tuple], group: int, constraint: str = "c1_c2",
                     temp: float = 0.8, max_tokens: int = 300, seed: int = 0, with_masks: bool = True,
                     reward_kind: str = "task") -> list[Rollout]:
    """`group` samples per prompt under `constraint` ('c1_c2' for SSD/RFT, 'none' for GRPO-only).
    prompts: (pid, prompt_text[, gold_mlir]). Generated ids get EOS appended when the sequence finished."""
    out = []
    grammar = gen.grammar("mlir_gen_c1c2") if with_masks else None
    eos = gen.tokenizer.eos_token_id
    for item in prompts:
        pid, prompt = item[0], item[1]; gold = item[2] if len(item) > 2 else None
        ids = gen.encode(prompt)
        for k in range(group):
            mx.random.seed(seed * 100003 + pid * 101 + k)
            g = gen._run(ids, constraint, max_tokens, temp)
            if not g.tokens:
                continue
            gen_ids = g.tokens + ([eos] if g.finished else [])
            bm = None
            if with_masks:
                try:
                    bm = replay_allowed_sets(gen.llg_tok, grammar, gen_ids, eos_id=eos)
                except ValueError:
                    bm = None  # sequence left the grammar (only possible for unmasked rollouts)
            r, parts = reward_fn(g.text, gold, reward_kind)
            out.append(Rollout(pid, ids, gen_ids, g.text, bm, r, parts))
    return out


def _gen_logits(model: nn.Module, prompt_ids: list[int], gen_ids: list[int]) -> mx.array:
    """Logits at the positions that predict gen_ids: shape (T, V), float32."""
    x = mx.array(prompt_ids + gen_ids)[None]
    logits = model(x)[0]
    P = len(prompt_ids)
    return logits[P - 1:P - 1 + len(gen_ids)].astype(mx.float32)


def _log_softmax(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def legal_mass_loss(model: nn.Module, ro: Rollout, vocab_size: int) -> mx.array:
    logits = _gen_logits(model, ro.prompt_ids, ro.gen_ids)
    allowed = mx.array(np.stack([bitmask_to_bool(r, vocab_size) for r in ro.bitmask]))  # (T, V) bool
    lp = _log_softmax(logits)
    logz = mx.logsumexp(mx.where(allowed, lp, mx.array(-1e30, dtype=lp.dtype)), axis=-1)  # log legal mass
    return -mx.mean(logz)


def seq_logprob_per_token(model: nn.Module, ro: Rollout) -> mx.array:
    logits = _gen_logits(model, ro.prompt_ids, ro.gen_ids)
    lp = _log_softmax(logits)
    tgt = mx.array(ro.gen_ids)
    return mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]  # (T,)


def teacher_logprobs(ref_model: nn.Module, ro: Rollout, vocab_size: int, masked: bool = True) -> np.ndarray:
    """log π_ref^mask(x_t | x_<t) for the rollout's tokens under the frozen reference model, renormalized
    over the allowed set at each step when `masked` and a bitmask is available."""
    logits = _gen_logits(ref_model, ro.prompt_ids, ro.gen_ids)
    lp = _log_softmax(logits)
    tgt = mx.array(ro.gen_ids)
    tok_lp = mx.take_along_axis(lp, tgt[:, None], axis=-1)[:, 0]
    if masked and ro.bitmask is not None:
        allowed = mx.array(np.stack([bitmask_to_bool(r, vocab_size) for r in ro.bitmask]))
        logz = mx.logsumexp(mx.where(allowed, lp, mx.array(-1e30, dtype=lp.dtype)), axis=-1)
        tok_lp = tok_lp - logz
    return np.array(tok_lp.astype(mx.float32))


def sft_loss(model: nn.Module, ro: Rollout) -> mx.array:
    return -mx.mean(seq_logprob_per_token(model, ro))


def group_advantages(rollouts: list[Rollout]) -> dict[int, float]:
    """GRPO advantage per rollout index: (r − mean_group) / (std_group + eps), groups by prompt_id."""
    by_pid: dict[int, list[int]] = {}
    for i, ro in enumerate(rollouts):
        by_pid.setdefault(ro.prompt_id, []).append(i)
    adv = {}
    for pid, idxs in by_pid.items():
        rs = np.array([rollouts[i].reward for i in idxs], dtype=np.float64)
        mu, sd = rs.mean(), rs.std()
        for i in idxs:
            adv[i] = float((rollouts[i].reward - mu) / (sd + 1e-6)) if sd > 0 else 0.0
    return adv


def grpo_loss(model: nn.Module, ro: Rollout, advantage: float, beta: float = 0.0, clip: float = 5.0) -> mx.array:
    """−mean_t A_t · log π_θ(x_t), A_t = advantage − β·stopgrad(log π_θ(x_t) − log π_teacher(x_t)).
    With β>0 and a fixed teacher this is on-policy distillation's implicit per-token reward added to the
    group-relative task advantage."""
    lp = seq_logprob_per_token(model, ro)
    if beta > 0 and ro.teacher_lp is not None:
        ratio = mx.clip(mx.stop_gradient(lp) - mx.array(ro.teacher_lp), -clip, clip)
        adv = advantage - beta * ratio
    else:
        if advantage == 0.0:
            return mx.array(0.0)
        adv = mx.array(advantage)
    return -mx.mean(adv * lp)


@dataclass
class StepStats:
    loss: float
    loss_mask: float
    loss_grpo: float
    n_rollouts: int
    mean_reward: float
    mean_legal_mass: float  # exp(mean log Z) before the step
    dt: float


def ssd_step(model: nn.Module, optimizer, rollouts: list[Rollout], vocab_size: int, lam: float = 1.0,
             mode: str = "ssd", micro_batch: int = 4, beta: float = 0.0) -> StepStats:
    """One optimizer step over `rollouts`. mode ∈ {ssd, rft, grpo, sft}.
    ssd: legal-mass + λ·GRPO; rft/sft: SFT on verifier-passing rollouts (rft) or all given (sft); grpo: reward only."""
    t0 = time.perf_counter()
    adv = group_advantages(rollouts) if mode in ("ssd", "grpo") else {}
    if mode == "rft":
        rollouts = [r for r in rollouts if r.reward > 0]
    if not rollouts:
        return StepStats(0.0, 0.0, 0.0, 0, 0.0, 0.0, time.perf_counter() - t0)

    def loss_fn(model, batch_idx):
        total = mx.array(0.0); lm = mx.array(0.0); lg = mx.array(0.0)
        for i in batch_idx:
            ro = rollouts[i]
            if mode == "ssd":
                if ro.bitmask is not None:
                    l1 = legal_mass_loss(model, ro, vocab_size); lm = lm + l1; total = total + l1
                if lam > 0 or beta > 0:
                    l2 = grpo_loss(model, ro, lam * adv.get(i, 0.0), beta); lg = lg + l2; total = total + l2
            elif mode == "grpo":
                l2 = grpo_loss(model, ro, adv.get(i, 0.0), beta); lg = lg + l2; total = total + l2
            else:
                total = total + sft_loss(model, ro)
        return total / len(batch_idx), (lm / len(batch_idx), lg / len(batch_idx))

    vg = nn.value_and_grad(model, loss_fn)
    n = len(rollouts); acc_grads = None; tot = lm_tot = lg_tot = 0.0; n_mb = 0
    for s in range(0, n, micro_batch):
        idx = list(range(s, min(n, s + micro_batch)))
        (loss, (lm, lg)), grads = vg(model, idx)
        scale = len(idx) / n
        grads = mx.tree_map(lambda g: g * scale, grads) if hasattr(mx, "tree_map") else _tree_scale(grads, scale)
        acc_grads = grads if acc_grads is None else _tree_add(acc_grads, grads)
        mx.eval(acc_grads)
        tot += float(loss) * scale; lm_tot += float(lm) * scale; lg_tot += float(lg) * scale; n_mb += 1
    optimizer.update(model, acc_grads)
    mx.eval(model.parameters(), optimizer.state)
    return StepStats(tot, lm_tot, lg_tot, n, float(np.mean([r.reward for r in rollouts])),
                     float(np.exp(-lm_tot)) if mode == "ssd" else float("nan"), time.perf_counter() - t0)


def _tree_scale(tree, s):
    from mlx.utils import tree_map
    return tree_map(lambda g: g * s, tree)


def _tree_add(a, b):
    from mlx.utils import tree_map
    return tree_map(lambda x, y: x + y, a, b)


def make_lora(model: nn.Module, rank: int = 32, num_layers: int | None = None) -> nn.Module:
    from mlx_lm.tuner.utils import linear_to_lora_layers
    n = num_layers or len(model.layers)
    model.freeze()
    linear_to_lora_layers(model, n, {"rank": rank, "scale": 20.0, "dropout": 0.0,
                                     "keys": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]})
    return model


def collect_rollouts_batched(gen: Generator, prompts: list[tuple], group: int, constraint: str = "c1_c2",
                             temp: float = 0.8, max_tokens: int = 300, seed: int = 0, with_masks: bool = True,
                             completion_batch_size: int = 64, reward_kind: str = "task") -> list[Rollout]:
    """Same contract as `collect_rollouts` but uses mlx_lm's continuous-batching generator with one
    grammar processor per sequence. Sequences whose grammar has accepted are removed early."""
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    from sci.masks.llg import GrammarLogitsProcessor

    grammar = gen.grammar("mlir_gen_c1c2")
    mx.random.seed(seed)
    bg = BatchGenerator(gen.model, max_tokens=max_tokens, stop_tokens=[[t] for t in gen.eos_ids],
                        sampler=make_sampler(temp=temp), completion_batch_size=completion_batch_size, prefill_batch_size=8)
    eos = gen.tokenizer.eos_token_id
    seqs = []  # (pid, prompt_ids, processor, gold)
    for item in prompts:
        pid, prompt = item[0], item[1]; gold = item[2] if len(item) > 2 else None
        ids = gen.encode(prompt)
        for _ in range(group):
            proc = GrammarLogitsProcessor(gen.llg_tok, grammar) if constraint != "none" else None
            seqs.append((pid, ids, proc, gold))
    uids = bg.insert([s[1] for s in seqs], [max_tokens] * len(seqs),
                     logits_processors=[[s[2]] if s[2] is not None else [] for s in seqs])
    by_uid = {uid: i for i, uid in enumerate(uids)}
    toks: dict[int, list[int]] = {uid: [] for uid in uids}
    finished: dict[int, bool] = {uid: False for uid in uids}
    done: set[int] = set()
    while len(done) < len(uids):
        responses = bg.next_generated()
        if not responses:
            break
        to_remove = []
        for r in responses:
            if r.uid in done:
                continue
            if r.finish_reason != "stop":
                toks[r.uid].append(r.token)
            if r.finish_reason is not None:
                finished[r.uid] = (r.finish_reason == "stop"); done.add(r.uid)
                continue
            proc = seqs[by_uid[r.uid]][2]
            if proc is not None and proc.matcher.is_stopped():
                # grammar accepted (or errored): stop this sequence; drop the token emitted after acceptance
                finished[r.uid] = not proc.matcher.is_error(); done.add(r.uid); to_remove.append(r.uid)
        if to_remove:
            bg.remove(to_remove)
    bg.close()
    pending = []
    for uid in uids:
        pid, ids, proc, gold = seqs[by_uid[uid]]
        gen_ids = toks[uid]
        if proc is not None:
            gen_ids = gen_ids[: proc.consumed] if proc.consumed and proc.consumed <= len(gen_ids) else gen_ids
        if not gen_ids:
            continue
        text = gen.tokenizer.decode(gen_ids)
        if finished[uid]:
            gen_ids = gen_ids + [eos]
        bm = None
        if with_masks:
            try:
                bm = replay_allowed_sets(gen.llg_tok, grammar, gen_ids, eos_id=eos)
            except ValueError:
                bm = None
        pending.append((pid, ids, gen_ids, text, bm, gold))
    rewards = rewards_parallel([p[3] for p in pending], [p[5] for p in pending], reward_kind)
    return [Rollout(pid, ids, gen_ids, text, bm, rw, parts) for (pid, ids, gen_ids, text, bm, gold), (rw, parts) in zip(pending, rewards)]
