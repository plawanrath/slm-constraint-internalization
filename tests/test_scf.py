"""Tests for the scf transfer target (grammars, golds, region-aware C3, glue).

Markers: `mlx` for the tokenizer replay (needs llguidance + the SmolLM2 tokenizer in
the HF cache), `docker` for mlir-opt / mlir-cpu-runner in the sci-llvm container.
"""
from __future__ import annotations

import json
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

from sci.transfer.scf.grammar import (GRAMMAR_NAMES, LATTICE_PATH, grammar_text, is_parse_valid_scf,
                                      llg_grammar)
from sci.transfer.scf.scope import accept_or_reject, validate
from sci.transfer.scf.task import build_prompt, chatml_prompt

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "data/benchmarks/scf_spec_60.jsonl"
REFS = REPO / "data/functional/scf_references.jsonl"
MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def golds() -> list[dict]:
    rows = _rows(BENCH)
    assert len(rows) == 60
    return rows


@lru_cache(maxsize=None)
def _gen_parser(name: str):
    from lark import Lark
    return Lark(grammar_text(name), parser="earley", lexer="dynamic_complete")


def _gen_accepts(name: str, text: str) -> bool:
    from lark.exceptions import LarkError
    try:
        _gen_parser(name).parse(text)
        return True
    except LarkError:
        return False


# ---------------- grammars ----------------

@pytest.mark.parametrize("name", GRAMMAR_NAMES)
def test_gen_grammar_compiles_under_llguidance(name):
    assert llg_grammar(name)


def test_benchmark_rows_schema(golds):
    ids = [r["id"] for r in golds]
    assert len(set(ids)) == 60 and [r["prompt_id"] for r in golds] == list(range(60))
    for r in golds:
        assert r["dialect"] == "scf" and r["nl"] and r["gold"] == r["mlir"]
        assert "scf." in r["mlir"]


@pytest.mark.parametrize("name", GRAMMAR_NAMES)
def test_gen_grammar_accepts_all_golds(name, golds):
    rejected = [r["id"] for r in golds if not _gen_accepts(name, r["mlir"])]
    assert not rejected, rejected


def test_parse_grammar_accepts_all_golds(golds):
    rejected = [r["id"] for r in golds if not is_parse_valid_scf(r["mlir"])]
    assert not rejected, rejected


def test_parse_grammar_still_accepts_training_dialect_and_rejects_junk():
    assert is_parse_valid_scf("module {\n  func.func @f(%a: i32) -> i32 {\n    %0 = arith.addi %a, %a : i32\n    return %0 : i32\n  }\n}")
    assert not is_parse_valid_scf("module { func.func @f() { scf.for { } return } }")


def test_gen_grammar_rejects_c2_domain_violations():
    bad = """module {
  func.func @f(%n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %z = arith.constant 0 : i32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%a = %z) -> (i32) {
      scf.yield %a : f32
    }
    return %r : i32
  }
}
"""
    assert not _gen_accepts("mlir_gen_scf", bad)      # yield domain must match the result domain
    assert _gen_accepts("mlir_gen_scf_c1", bad)       # C1-only grammar has no domain split
    assert not _gen_accepts("mlir_gen_scf", bad.replace("scf.yield %a : f32", "%q = arith.addf %a, %a : i32\n      scf.yield %q : i32"))


def test_lattice_has_region_ops():
    lat = json.loads(LATTICE_PATH.read_text())
    assert {"scf.for", "scf.if", "scf.yield"} <= set(lat)
    assert lat["scf.if"]["operand_types"] == [["i1"]]
    assert lat["scf.yield"]["result_arity"] == [0, 0]


@pytest.mark.mlx
@pytest.mark.parametrize("name", GRAMMAR_NAMES)
def test_golds_replay_under_tokenizer_mask(name, golds):
    pytest.importorskip("mlx.core")
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    from sci.masks.llg import llg_tokenizer, replay_allowed_sets
    hf = load_tokenizer(Path(snapshot_download(MODEL, allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf)
    g = llg_grammar(name)
    for r in golds:
        ids = hf.encode(r["mlir"], add_special_tokens=False) + [hf.eos_token_id]
        masks = replay_allowed_sets(tok, g, ids, eos_id=hf.eos_token_id)
        assert masks.shape[0] == len(ids)


# ---------------- C3 validator ----------------

def test_validator_accepts_all_golds(golds):
    rejected = {r["id"]: [str(v) for v in validate(r["mlir"]).violations] for r in golds
                if not validate(r["mlir"]).passed}
    assert not rejected, rejected


GOOD = textwrap.dedent("""\
    module {
      func.func @sum(%n: index, %m: memref<?xi32>) -> i32 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %z = arith.constant 0 : i32
        %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) -> (i32) {
          %ii = arith.index_cast %i : index to i32
          %s = arith.addi %acc, %ii : i32
          scf.yield %s : i32
        }
        %c = arith.cmpi sgt, %r, %z : i32
        %x = scf.if %c -> (i32) {
          scf.yield %r : i32
        } else {
          scf.yield %z : i32
        }
        scf.if %c {
          memref.store %x, %m[%c0] : memref<?xi32>
        }
        return %x : i32
      }
    }
""")


def _kinds(src: str) -> set[str]:
    return {v.kind for v in validate(src).violations}


def test_validator_accepts_good_and_one_line_form():
    assert accept_or_reject(GOOD)[0]
    assert accept_or_reject(GOOD.replace("\n", " "))[0]


def test_loop_index_after_loop_is_undefined():
    src = GOOD.replace("return %x : i32", "%q = arith.index_cast %i : index to i32\n    return %q : i32")
    assert "undef_use" in _kinds(src)


def test_inner_region_value_dead_outside():
    assert "undef_use" in _kinds(GOOD.replace("return %x : i32", "return %s : i32"))


def test_yield_type_mismatch():
    src = GOOD.replace("scf.yield %s : i32", "%w = arith.extsi %s : i32 to i64\n      scf.yield %w : i64")
    assert "yield_mismatch" in _kinds(src)
    assert "yield_mismatch" in _kinds(GOOD.replace("scf.yield %r : i32", "scf.yield"))


def test_iter_arg_used_before_loop():
    src = GOOD.replace("%z = arith.constant 0 : i32", "%z = arith.addi %acc, %acc : i32")
    assert "undef_use" in _kinds(src)


def test_region_results_defined_after_op_and_if_requires_else():
    assert "yield_mismatch" in _kinds(GOOD.replace("} else {\n      scf.yield %z : i32\n    }", "}"))
    assert "type_mismatch" in _kinds(GOOD.replace("scf.if %c {", "scf.if %r {"))
    reuse = GOOD.replace("return %x : i32", "%t = arith.addi %x, %r : i32\n    return %t : i32")
    assert accept_or_reject(reuse)[0]


def test_two_results_order_checked():
    src = textwrap.dedent("""\
        module {
          func.func @f(%n: index) -> i32 {
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %z = arith.constant 0 : i32
            %o = arith.constant 1.0 : f32
            %s, %p = scf.for %i = %c0 to %n step %c1 iter_args(%a = %z, %b = %o) -> (i32, f32) {
              %a2 = arith.addi %a, %a : i32
              %b2 = arith.mulf %b, %o : f32
              scf.yield %a2, %b2 : i32, f32
            }
            %pi = arith.fptosi %p : f32 to i32
            %t = arith.addi %s, %pi : i32
            return %t : i32
          }
        }
    """)
    assert accept_or_reject(src)[0]
    assert "yield_mismatch" in _kinds(src.replace("scf.yield %a2, %b2 : i32, f32", "scf.yield %b2, %a2 : f32, i32"))
    assert "type_mismatch" in _kinds(src.replace("%pi = arith.fptosi %p : f32 to i32", "%pi = arith.fptosi %s : f32 to i32"))


def test_validator_abstains_on_unknown_ops_and_no_function():
    assert accept_or_reject("scf.yield")[0]
    src = GOOD.replace("%ii = arith.index_cast %i : index to i32", "%ii = math.absi %i : i32")
    assert accept_or_reject(src)[0]


# ---------------- prompt builder / references / registrations ----------------

def test_prompt_builder_fallback_and_template():
    class NoTemplate:
        chat_template = None

    p = build_prompt(NoTemplate(), "Write a function `f` that sums a memref.")
    assert p == chatml_prompt("Write a function `f` that sums a memref.")
    assert p.count("Example") == 3 and p.endswith("MLIR:<|im_end|>\n<|im_start|>assistant\n")

    class Template:
        chat_template = "x"

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking=False):
            return "|".join(m["role"] + ":" + m["content"][:8] for m in msgs)

    assert build_prompt(Template(), "task") == "system:Output o|user:Example "


def test_references_align_with_benchmark(golds):
    refs = _rows(REFS)
    assert len(refs) == 60
    by_id = {r["id"]: r for r in golds}
    for ref in refs:
        g = by_id[ref["source_id"]]
        assert ref["prompt_id"] == g["prompt_id"] and ref["dialect"] == "scf"
        assert f"@{ref['canonical_fn_name']}(" in g["mlir"]
        assert len(ref["input_domains"]) == len(ref["signature"]["args"])


def test_functional_module_registers_scf_target():
    import sci.eval.differential as D
    import sci.eval.functional as Fn
    from sci.transfer.scf import functional as F
    assert D.DIALECTS["scf"]["cand_dialect"] == "scf" and Fn.DISPATCH["scf"] is F.run_scf
    assert "--convert-scf-to-cf" in F.SCF_PIPELINE
    ref = _rows(REFS)[0]
    sig = D.parse_first_fn(F.gold_for(ref))
    td = F.trial_inputs(sig, ref, "scf:test", 0)
    assert td == F.trial_inputs(sig, ref, "scf:test", 0)
    assert all(1 <= v <= 7 for v, t in zip(td["values"], sig["arg_types"]) if t == "index")


def test_ladder_runner_resumes_and_summarizes(tmp_path, golds):
    from sci.transfer.scf.ladder import Cell, Target, load_set, residuals, run_cell, write_summary

    class Out:
        def __init__(self, text):
            self.text, self.tokens, self.attempts, self.scope_passed_per_try, self.finished, self.dt = text, [1, 2], 1, [], True, 0.01

    calls = []

    class Stub:
        tokenizer = None

        def generate(self, prompt, constraint, max_tokens, seed):
            calls.append(constraint)
            return Out(golds[len(calls) % 2]["mlir"] if constraint != "none" else "garbage")

    target = Target(dialect="scf", make_generator=lambda m, r: Stub(), build_prompt=lambda t, nl: nl,
                    parse_valid=is_parse_valid_scf, scope_ok=lambda t: accept_or_reject(t)[0],
                    verify=lambda t: {"returncode": 0, "stderr": ""})
    pools = {"scf_spec_60": load_set(BENCH)}
    out = tmp_path / "ladder.jsonl"
    cell = Cell(tag="stub", model="m")
    run_cell(cell, target, pools, ("none", "c1_c2_c3"), out, limit=3)
    assert len(calls) == 6
    run_cell(cell, target, pools, ("none", "c1_c2_c3"), out, limit=3)   # resumable: nothing new
    assert len(calls) == 6
    rows = _rows(out)
    assert len(rows) == 6 and all(r["dialect"] == "scf" for r in rows)
    res = residuals(rows)
    assert res["rates"]["none"]["v1"] == 1.0 and res["rates"]["c1_c2_c3"]["verify_valid"] == 1.0
    assert res["residual_pp"]["v3"]["vs"] == "c1_c2_c3" and "v1" not in res["residual_pp"]  # v1 pairs with c1 (not run)
    summary = write_summary(out, tmp_path / "summary.json")
    assert summary["cells"]["stub"]["scf_spec_60"]["c1_c2_c3"]["verify_rate"] == 1.0


# ---------------- docker ----------------

@pytest.mark.docker
def test_all_golds_verify_under_mlir_opt(golds):
    from sci.eval.verify import verify
    failed = {r["id"]: verify(r["mlir"])["stderr"][:200] for r in golds if verify(r["mlir"])["returncode"] != 0}
    assert not failed, failed


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.parametrize("key", ["sum_to_n_i32", "nested_2d_sum_f32", "memref_prefix_sum"])
def test_golds_run_functionally_through_scf_lowering(key, golds):
    from sci.transfer.scf import functional as F
    ref = next(r for r in _rows(REFS) if r["id"].endswith(key))
    gold = F.gold_for(ref)
    gate = F.gold_gate(gold, ref)
    assert gate["ok"], gate["err"]
    assert F.differential(gold, gold, ref, gate=gate)["status"] == "match"
    import re
    broken = re.sub(r"arith\.add([if]) ", r"arith.mul\1 ", gold, count=1)   # first accumulate becomes a product
    assert F.differential(broken, gold, ref, gate=gate)["status"] == "mismatch"
