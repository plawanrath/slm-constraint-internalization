"""Batched training forward == per-rollout forward (needs MLX + the 135M checkpoint; marked mlx)."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def rollouts():
    from sci.eval.generate import Generator, build_prompt
    from sci.masks.llg import replay_allowed_sets
    from sci.train.spine import Rollout
    gen = Generator(MODEL)
    grammar = gen.grammar("mlir_gen_c1c2")
    eos = gen.tokenizer.eos_token_id
    nls = ["Write a function that multiplies two i32 values.", "Write a function that subtracts two f32 values and returns the result."]
    ros = []
    for pid, nl in enumerate(nls):
        ids = gen.encode(build_prompt(gen.tokenizer, nl, "arith+func"))
        for k in range(2):
            mx.random.seed(pid * 10 + k)
            g = gen._run(ids, "c1_c2", 60, 0.8)
            gen_ids = g.tokens + ([eos] if g.finished else [])
            bm = replay_allowed_sets(gen.llg_tok, grammar, gen_ids, eos_id=eos)
            ros.append(Rollout(pid, ids, gen_ids, g.text, bm, float(k)))  # rewards 0/1 within each group
    gen.model.set_dtype(mx.float32)  # bf16 batched matmuls differ from single-sequence ones by up to ~0.1 nat; fp32 by ~1e-5
    return gen, ros


@pytest.mark.mlx
def test_teacher_logprobs_batched_matches(rollouts):
    from sci.train.spine import teacher_logprobs, teacher_logprobs_batched
    gen, ros = rollouts
    V = gen.llg_tok.vocab_size
    seq = [teacher_logprobs(gen.model, ro, V, masked=True) for ro in ros]
    teacher_logprobs_batched(gen.model, ros, V, masked=True, micro_batch=4)
    for a, ro in zip(seq, ros):
        assert ro.teacher_lp.shape == a.shape
        np.testing.assert_allclose(ro.teacher_lp, a, atol=1e-3, rtol=0)


@pytest.mark.mlx
@pytest.mark.parametrize("mode", ["ssd", "grpo", "sft"])
def test_micro_batch_loss_matches(rollouts, mode):
    from sci.train.spine import group_advantages, micro_batch_loss, teacher_logprobs_batched
    gen, ros = rollouts
    V = gen.llg_tok.vocab_size
    teacher_logprobs_batched(gen.model, ros, V, masked=(mode == "ssd"))
    adv = group_advantages(ros)
    advs = [adv[i] for i in range(len(ros))]
    (tb, (lmb, lgb, _)) = micro_batch_loss(gen.model, ros, advs, V, mode=mode, lam=1.0, beta=0.1, batched=True)
    (ts, (lms, lgs, _)) = micro_batch_loss(gen.model, ros, advs, V, mode=mode, lam=1.0, beta=0.1, batched=False)
    mx.eval(tb, ts, lmb, lms, lgb, lgs)
    for b, s in ((tb, ts), (lmb, lms), (lgb, lgs)):
        assert abs(float(b) - float(s)) <= 1e-3 * max(1.0, abs(float(s))), (mode, float(b), float(s))


@pytest.mark.mlx
def test_c3_mass_loss_batched_matches(rollouts):
    from sci.masks.c3_offline import C3MaskBuilder
    from sci.train.spine import group_advantages, micro_batch_loss
    gen, ros = rollouts
    V = gen.llg_tok.vocab_size
    b = C3MaskBuilder(gen.tokenizer, V)
    for ro in ros:
        ro.c3_rows = b.build(ro.gen_ids)
    assert any(ro.c3_rows for ro in ros)
    adv = group_advantages(ros); advs = [adv[i] for i in range(len(ros))]
    (tb, (_, _, lcb)) = micro_batch_loss(gen.model, ros, advs, V, mode="ssd", lam=1.0, beta=0.0, batched=True, c3_weight=1.0)
    (ts, (_, _, lcs)) = micro_batch_loss(gen.model, ros, advs, V, mode="ssd", lam=1.0, beta=0.0, batched=False, c3_weight=1.0)
    mx.eval(tb, ts, lcb, lcs)
    assert float(lcb) > 0.0
    assert abs(float(lcb) - float(lcs)) <= 1e-3 * max(1.0, abs(float(lcs)))
    assert abs(float(tb) - float(ts)) <= 1e-3 * max(1.0, abs(float(ts)))
