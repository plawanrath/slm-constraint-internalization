"""LLVM IR transfer target: grammars, LLVM-Spec-60 golds, scope validator, verifier, harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sci.transfer.llvmir.functional import build_driver, gen_inputs, parse_define
from sci.transfer.llvmir.grammar import GRAMMAR_NAMES, grammar_text, is_parse_valid, llg_grammar
from sci.transfer.llvmir.scope import accept_or_reject, validate
from sci.transfer.llvmir.task import SYSTEM, build_prompt, chatml_prompt

REPO = Path(__file__).resolve().parents[1]
BENCH = [json.loads(l) for l in (REPO / "data/benchmarks/llvm_spec_60.jsonl").read_text().splitlines() if l.strip()]
REFS = {r["prompt_id"]: r for r in
        (json.loads(l) for l in (REPO / "data/functional/llvm_references.jsonl").read_text().splitlines() if l.strip())}
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


# ---------------- grammars ----------------

@pytest.mark.parametrize("name", GRAMMAR_NAMES)
def test_grammar_compiles_with_llguidance(name):
    from llguidance import LLMatcher
    spec = LLMatcher.grammar_from_lark(grammar_text(name))
    assert spec and llg_grammar(name) == spec


def test_benchmark_shape():
    assert len(BENCH) == 60
    assert [r["prompt_id"] for r in BENCH] == list(range(60))
    for r in BENCH:
        assert r["dialect"] == "llvmir" and r["nl"] and r["gold"]
        sig = parse_define(r["gold"])
        assert sig["arg_types"] == r["signature"]["args"] and sig["ret"] == r["signature"]["ret"]
        assert [d["type"] for d in REFS[r["prompt_id"]]["input_domains"]] == sig["arg_types"]
    families = {r["family"] for r in BENCH}
    assert {"arith", "farith", "cmp_select", "branch_phi", "loop", "memory", "mixed"} <= families


@pytest.mark.parametrize("row", BENCH, ids=[r["id"] for r in BENCH])
def test_gold_parses_under_both_grammars(row):
    assert is_parse_valid(row["gold"], "llvm_gen_c1")
    assert is_parse_valid(row["gold"], "llvm_gen_c1c2")


def test_c2_grammar_rejects_type_domain_errors():
    ok = BENCH[0]["gold"]
    for bad in (ok.replace("add i32", "fadd i32"), ok.replace("add i32 %a, %b", "add i32 %a, 1.5"),
                ok.replace("add i32 %a, %b", "icmp olt i32 %a, %b")):
        assert is_parse_valid(bad, "llvm_gen_c1")
        assert not is_parse_valid(bad, "llvm_gen_c1c2")


# ---------------- scope validator ----------------

@pytest.mark.parametrize("row", BENCH, ids=[r["id"] for r in BENCH])
def test_scope_accepts_gold(row):
    ok, rep = accept_or_reject(row["gold"])
    assert ok, [str(v) for v in rep.violations]
    assert rep.abstained == []


def _kinds(text: str) -> set[str]:
    return {v.kind for v in validate(text).violations}


def test_scope_rejects_use_before_def():
    src = "define i32 @f(i32 %a) {\nentry:\n  %r = add i32 %t, 1\n  %t = mul i32 %a, 2\n  ret i32 %r\n}\n"
    assert "use_before_def" in _kinds(src)


def test_scope_rejects_undefined_and_self_reference():
    assert "undef_use" in _kinds("define i32 @f(i32 %a) {\nentry:\n  ret i32 %zz\n}\n")
    assert "use_before_def" in _kinds("define i32 @f(i32 %a) {\nentry:\n  %x = add i32 %x, %a\n  ret i32 %x\n}\n")


def test_scope_rejects_non_dominating_use():
    src = ("define i32 @f(i32 %a, i1 %c) {\nentry:\n  br i1 %c, label %then, label %else\n"
           "then:\n  %x = add i32 %a, 1\n  br label %join\nelse:\n  br label %join\njoin:\n  ret i32 %x\n}\n")
    assert _kinds(src) == {"not_dominated"}


def test_scope_rejects_bad_phi():
    good = ("define i32 @f(i32 %a, i1 %c) {\nentry:\n  br i1 %c, label %then, label %else\n"
            "then:\n  %x = add i32 %a, 1\n  br label %join\nelse:\n  %y = sub i32 %a, 1\n  br label %join\n"
            "join:\n  %r = phi i32 [ %x, %then ], [ %y, %else ]\n  ret i32 %r\n}\n")
    assert accept_or_reject(good)[0]
    swapped = good.replace("[ %x, %then ], [ %y, %else ]", "[ %y, %then ], [ %x, %else ]")
    assert _kinds(swapped) == {"bad_phi"}
    missing = good.replace("[ %x, %then ], [ %y, %else ]", "[ %x, %then ]")
    assert "bad_phi" in _kinds(missing)
    not_first = good.replace("join:\n  %r = phi", "join:\n  %z = add i32 %a, 2\n  %r = phi")
    assert "bad_phi" in _kinds(not_first)


def test_scope_loop_backedge_phi_is_legal():
    loop = next(r for r in BENCH if r["id"].startswith("39_"))["gold"]
    assert accept_or_reject(loop)[0]
    # a value from the loop body used in the entry block does not dominate it
    broken = loop.replace("entry:\n  br label %loop", "entry:\n  %pre = add i32 %acc.next, 1\n  br label %loop")
    assert "not_dominated" in _kinds(broken)


def test_scope_abstains_on_unknown_instruction():
    ok, rep = accept_or_reject("define i32 @f(i32 %a) {\nentry:\n  %r = call i32 @g(i32 %a)\n  ret i32 %r\n}\n")
    assert ok and rep.abstained == ["%r = call i32 @g(i32 %a)"]


def test_scope_handles_multiline_instruction():
    src = "define i32 @f(i32 %a,\n i32 %b) {\nentry:\n  %r = add i32 %a,\n    %b\n  ret i32 %r\n}\n"
    assert accept_or_reject(src)[0]


# ---------------- harness + prompt (no docker) ----------------

def test_inputs_seeded_and_in_domain():
    ref = REFS[3]  # sdiv: divisor nonzero
    a = [gen_inputs(ref, t) for t in range(5)]
    assert a == [gen_inputs(ref, t) for t in range(5)]
    assert all(v[1] != 0 for v in a) and len({tuple(v) for v in a}) > 1


def test_driver_shape():
    sig = parse_define(BENCH[0]["gold"])
    d = build_driver(BENCH[0]["gold"], sig, [3, 4])
    assert "call i32 @f(i32 3, i32 4)" in d and "@printf" in d and "define i32 @main()" in d


def test_prompt_rendering():
    p = chatml_prompt("Add two i32 values.")
    assert p.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>") and p.endswith("LLVM IR:<|im_end|>\n<|im_start|>assistant\n")

    class NoTemplate:
        chat_template = None

    assert build_prompt(NoTemplate(), "Add two i32 values.") == p


# ---------------- mlx: token-level replay of every gold ----------------

@pytest.mark.mlx
@pytest.mark.slow
def test_golds_replay_through_llguidance():
    pytest.importorskip("mlx.core")
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    from sci.masks.llg import bitmask_to_bool, llg_tokenizer, replay_allowed_sets
    hf = load_tokenizer(Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf)
    for name in GRAMMAR_NAMES:
        g = llg_grammar(name)
        for r in BENCH:
            ids = hf.encode(r["gold"], add_special_tokens=False)
            masks = replay_allowed_sets(tok, g, ids)
            assert masks.shape[0] == len(ids), (name, r["id"])
            for t, tid in enumerate(ids):
                assert bitmask_to_bool(masks[t], tok.vocab_size)[tid], (name, r["id"], t)


# ---------------- docker: llvm-as + lli ----------------

@pytest.mark.docker
@pytest.mark.slow
def test_all_golds_pass_llvm_as():
    from sci.transfer.llvmir.verify import verify
    failures = {r["id"]: verify(r["gold"])["stderr"][:120] for r in BENCH if verify(r["gold"])["returncode"] != 0}
    assert failures == {}


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.parametrize("prefix", ["01_", "39_", "34_"])
def test_gold_matches_itself_under_lli(prefix):
    from sci.transfer.llvmir.functional import differential
    row = next(r for r in BENCH if r["id"].startswith(prefix))
    d = differential(row["gold"], row["gold"], REFS[row["prompt_id"]])
    assert d["status"] == "match" and d["n_trials_matched"] == 5, d
    mutant = row["gold"].replace("ret i32 %", "ret i32 0 ; %")
    assert differential(mutant, row["gold"], REFS[row["prompt_id"]])["status"] == "mismatch"
