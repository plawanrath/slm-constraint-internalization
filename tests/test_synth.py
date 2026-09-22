"""Synthetic SSA language: generator, checker, grammars, task adapter, and the law fit."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sci.synth import checker
from sci.synth.generator import generate, make_pool
from sci.synth.grammar import GRAMMAR_NAMES, grammar_path, grammar_text, render_grammar
from sci.synth.law import aggregate_seeds, fit_power_law, predict, residual_from_rows
from sci.synth.task import SynthTask, chatml_prompt

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
GRID = [(D, N, nest) for D in (2, 4, 8, 16, 32, 64) for N in (1, 3, 6) for nest in (0, 1, 2)]

GOOD = """prog {
  %v0 = const 3 : int[4]
  %v1 = const 1.5 : flt[2x3]
  {
    %v2 = addi %v0 , %v0 : int[4]
  }
  %v3 = subi %v0 , %v0 : int[4]
  ret %v3 : int[4]
}
"""


# ---------------- generator ----------------

def test_generate_deterministic():
    a = generate(8, 3, 1, seed=5, n_ops=20)
    b = generate(8, 3, 1, seed=5, n_ops=20)
    assert a == b
    assert generate(8, 3, 1, seed=6, n_ops=20) != a


@pytest.mark.parametrize("D,N,nest", GRID)
def test_gold_valid_and_distance_equals_D(D, N, nest):
    for seed in range(3):
        nl, gold = generate(D, N, nest, seed, n_ops=D + N + 6 if seed else None)
        rep = checker.check(gold)
        assert rep.passed, (D, N, nest, seed, rep.as_dict())
        assert checker.max_dep_distance(gold) == D
        assert nl and str(D) in nl


def test_probe_distance_is_exact():
    nl, gold = generate(16, 4, 2, seed=1)
    ds = checker.dep_distances(gold)
    assert max(d for *_, d in ds) == 16
    assert nl.count("exactly 16 statements") == 1


def test_nesting_produces_blocks():
    assert all("{" not in generate(8, 4, 0, s)[1].split("\n", 1)[1] for s in range(5))
    assert any("  {" in generate(8, 4, 1, s)[1] for s in range(20))


def test_make_pool_rows(tmp_path):
    p = tmp_path / "pool.jsonl"
    rows = make_pool(5, 4, 2, 1, seed=3, path=p)
    assert [r["prompt_id"] for r in rows] == list(range(5))
    back = [json.loads(l) for l in p.read_text().splitlines()]
    assert back == rows
    assert set(rows[0]) == {"prompt_id", "nl", "gold", "D", "N", "nesting", "seed"}
    assert all(checker.check(r["gold"]).passed for r in rows)


# ---------------- checker ----------------

def test_checker_accepts_good():
    rep = checker.check(GOOD)
    assert rep.passed and rep.as_dict()["violations"] == []
    assert checker.accept_or_reject(GOOD)[0]
    assert checker.scope_ok(GOOD)


def test_checker_c1_parse_error():
    rep = checker.check(GOOD.replace("ret %v3 : int[4]\n", ""))
    assert not rep.parse_valid and rep.violations[0].layer == 1
    rep = checker.check("prog {\n  %v0 = const 3 : int[4]\n  ret %v0 int[4]\n}\n")
    assert not rep.parse_valid


def test_checker_c2_type_domain():
    rep = checker.check(GOOD.replace("%v3 = subi %v0 , %v0 : int[4]", "%v3 = subf %v0 , %v0 : int[4]"))
    assert rep.parse_valid and not rep.type_ok and rep.scope_ok and rep.shape_ok
    assert {v.kind for v in rep.violations} == {"type_domain"}
    rep = checker.check(GOOD.replace("const 3 : int[4]", "const 3.0 : int[4]"))
    assert not rep.type_ok and {v.kind for v in rep.violations} == {"literal"}


def test_checker_c3_scope():
    # use before def
    rep = checker.check(GOOD.replace("%v3 = subi %v0 , %v0", "%v3 = subi %v0 , %v9"))
    assert rep.parse_valid and rep.type_ok and not rep.scope_ok and rep.shape_ok
    assert {v.kind for v in rep.violations} == {"undef_use"}
    # block-local name used after the block closes
    rep = checker.check(GOOD.replace("%v3 = subi %v0 , %v0", "%v3 = subi %v0 , %v2"))
    assert not rep.scope_ok and {v.kind for v in rep.violations} == {"dead_use"}
    # redefinition while in scope
    rep = checker.check(GOOD.replace("%v3 = subi", "%v0 = subi").replace("ret %v3", "ret %v0"))
    assert not rep.scope_ok and {v.kind for v in rep.violations} == {"redef"}
    # cross-statement domain mismatch counts as C3 (type_ssa), not C2 or C4
    rep = checker.check(GOOD.replace("%v3 = subi %v0 , %v0 : int[4]", "%v3 = addf %v1 , %v0 : flt[2x3]")
                        .replace("ret %v3 : int[4]", "ret %v3 : flt[2x3]"))
    assert rep.type_ok and not rep.scope_ok and rep.shape_ok
    assert {v.kind for v in rep.violations} == {"type_mismatch"}
    assert not checker.accept_or_reject(GOOD.replace("%v0 , %v0 : int[4]\n  ret", "%v0 , %v7 : int[4]\n  ret"))[0]


def test_checker_c4_shape():
    rep = checker.check(GOOD.replace("%v3 = subi %v0 , %v0 : int[4]", "%v3 = subi %v0 , %v0 : int[8]")
                        .replace("ret %v3 : int[4]", "ret %v3 : int[8]"))
    assert rep.parse_valid and rep.type_ok and rep.scope_ok and not rep.shape_ok
    assert {v.kind for v in rep.violations} == {"shape_mismatch"}
    # C4 is reported but not filtered by the C3 rejection hook
    assert checker.accept_or_reject(GOOD.replace(": int[4]\n  ret %v3 : int[4]", ": int[8]\n  ret %v3 : int[8]"))[0]


def test_checker_tolerates_grammar_whitespace():
    one_line = "prog { %v0 = const 3 : int[4] %v1 = addi %v0 , %v0 : int[4] ret %v1 : int[4] }"
    assert checker.check(one_line).passed


def test_reward_task_conditional():
    nl, gold = generate(4, 2, 0, seed=0)
    r, parts = checker.reward_fn(gold, gold)
    assert r == 1.0 and parts["verify"] and parts["scope"] and parts["parse"]
    assert checker.reward_fn("prog { garbage }", gold)[0] == 0.0
    # valid but structurally different program scores strictly between 0 and 1
    other = GOOD.replace("int[4]", "flt[2x3]").replace("const 3 :", "const 3.0 :").replace("addi", "addf").replace("subi", "mulf")
    assert checker.check(other).passed
    r2, p2 = checker.reward_fn(other, gold)
    assert 0.0 <= r2 < 1.0 and p2["verify"]
    assert checker.reward_fn(other, gold, kind="verify")[0] == 1.0
    assert checker.signature_task_reward(gold, gold) == 1.0


# ---------------- grammars ----------------

def test_grammar_files_match_render():
    assert grammar_text("ssa_gen") == render_grammar(typed=True)
    assert grammar_text("ssa_gen_c1") == render_grammar(typed=False)
    rules = [l for l in grammar_text("ssa_gen").splitlines() if not l.startswith("//")]
    assert not any("%ignore" in l for l in rules)
    with pytest.raises(KeyError):
        grammar_path("nope")


@pytest.mark.parametrize("name", GRAMMAR_NAMES)
def test_grammar_compiles_with_llguidance(name):
    LLMatcher = pytest.importorskip("llguidance").LLMatcher
    g = LLMatcher.grammar_from_lark(grammar_text(name))
    assert LLMatcher.validate_grammar(g) == ""


def test_lark_parses_golds_and_c2_grammar_rejects_type_error():
    lark = pytest.importorskip("lark")
    typed = lark.Lark(grammar_text("ssa_gen"), parser="earley")
    untyped = lark.Lark(grammar_text("ssa_gen_c1"), parser="earley")
    for D, N, nest in ((2, 1, 0), (8, 3, 1), (32, 6, 2)):
        gold = generate(D, N, nest, seed=0, n_ops=D + N + 4)[1]
        typed.parse(gold); untyped.parse(gold)
    bad = GOOD.replace("subi %v0 , %v0 : int[4]", "subf %v0 , %v0 : int[4]")
    untyped.parse(bad)
    with pytest.raises(lark.exceptions.LarkError):
        typed.parse(bad)


def test_grammar_max_depth_and_ret_last():
    lark = pytest.importorskip("lark")
    typed = lark.Lark(grammar_text("ssa_gen"), parser="earley")
    deep = "prog {\n  {\n    {\n      {\n        %v0 = const 1 : int[4]\n      }\n    }\n  }\n  ret %v0 : int[4]\n}\n"
    with pytest.raises(lark.exceptions.LarkError):
        typed.parse(deep)
    with pytest.raises(lark.exceptions.LarkError):
        typed.parse("prog {\n  ret %v0 : int[4]\n  %v0 = const 1 : int[4]\n}\n")


# ---------------- task adapter ----------------

def test_task_prompt_and_hooks():
    t = SynthTask()
    p = t.build_prompt(object(), "do the thing")
    assert p == chatml_prompt("do the thing") and "Example 3" in p and p.endswith("Prog:<|im_end|>\n<|im_start|>assistant\n")
    assert t.grammar_name == "ssa_gen" and t.gold_key == "gold"
    assert "prog" in t.grammar_lark()
    assert t.scope_ok(GOOD) and not t.scope_ok("prog {")
    assert t.reward(GOOD, GOOD)[0] == 1.0
    for ex in t.build_prompt(object(), "x").split("Prog:\n")[1:]:
        prog = ex.split("\n\nExample")[0].split("\n\nNow this task")[0]
        assert checker.check(prog).passed, prog


# ---------------- llguidance replay over the real tokenizer (needs MLX) ----------------

@pytest.mark.mlx
def test_golds_accepted_by_llguidance_mask():
    pytest.importorskip("mlx.core")
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    from sci.masks.llg import bitmask_to_bool, llg_tokenizer, replay_allowed_sets
    from sci.synth.grammar import compile_grammar
    hf = load_tokenizer(Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf)
    eos = hf.eos_token_id
    for name in GRAMMAR_NAMES:
        g = compile_grammar(name)
        for D, N, nest in ((2, 1, 0), (8, 4, 1), (64, 4, 2), (16, 8, 2)):
            gold = generate(D, N, nest, seed=D, n_ops=D + N + 5)[1]
            ids = hf.encode(gold, add_special_tokens=False) + [eos]
            masks = replay_allowed_sets(tok, g, ids, eos_id=eos)
            assert masks.shape[0] == len(ids)
            for t, tid in enumerate(ids):
                assert bitmask_to_bool(masks[t], tok.vocab_size)[tid], (name, D, t, hf.decode([tid]))
    with pytest.raises(ValueError):
        replay_allowed_sets(tok, compile_grammar("ssa_gen"), hf.encode(GOOD.replace("subi %v0 , %v0 : int[4]", "subf %v0 , %v0 : int[4]"), add_special_tokens=False))


# ---------------- residual and law fit ----------------

def test_residual_from_rows_paired():
    rows = [{"scope_ok_free": i % 4 != 0, "scope_ok_masked": i % 10 != 0} for i in range(200)]
    r = residual_from_rows(rows, n_resamples=2000)
    assert abs(r["v3_free"] - 0.25) < 1e-9 and abs(r["v3_masked"] - 0.10) < 1e-9
    assert abs(r["residual"] - 0.15) < 1e-9 and r["ci_low"] <= 0.15 <= r["ci_high"] and r["n"] == 200


def test_fit_recovers_known_exponents():
    a, alpha, beta = 0.3, 0.6, 0.25
    pts = [{"D": D, "size": s, "residual": a * D ** alpha * s ** (-beta)}
           for D in (2, 4, 8, 16, 32, 64) for s in (135e6, 360e6, 1.7e9)]
    fit = fit_power_law(pts)
    assert abs(fit["alpha"] - alpha) < 1e-9 and abs(fit["beta"] - beta) < 1e-9 and abs(fit["a"] - a) < 1e-9
    assert fit["r2"] > 0.999999 and fit["n_clipped"] == 0
    assert abs(predict(fit, 8, 360e6) - a * 8 ** alpha * 360e6 ** (-beta)) < 1e-12
    rng = np.random.default_rng(0)
    noisy = [{**p, "residual": p["residual"] * float(np.exp(rng.normal(0, 0.05)))} for p in pts]
    fit_n = fit_power_law(noisy)
    assert abs(fit_n["alpha"] - alpha) < 0.05 and abs(fit_n["beta"] - beta) < 0.05
    with pytest.raises(ValueError):
        fit_power_law(pts[:2])


def test_aggregate_seeds_and_clipping():
    rows = [{"model": "SmolLM2-135M-Instruct", "D": 4, "seed": s, "residual": 0.1 * (s + 1)} for s in range(3)]
    rows += [{"model": "SmolLM2-360M-Instruct", "D": 4, "seed": 0, "residual": 0.0}]
    pts = aggregate_seeds(rows)
    assert len(pts) == 2 and abs(pts[0]["residual"] - 0.2) < 1e-12 and pts[0]["n_seeds"] == 3 and pts[0]["size"] == 135e6
    pts += [{"D": 16, "size": 135e6, "residual": 0.4}]
    assert fit_power_law(pts)["n_clipped"] == 1
