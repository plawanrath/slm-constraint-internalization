"""Coverage template families (pool v1c): every new family draws parse-valid golds
that the C1+C2 generation grammar accepts (llguidance replay; `mlx`) and `mlir-opt` verifies (`docker`),
and the v1 family set still regenerates the released v1 pool row for row."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sci.constraints.parser import is_parse_valid
from sci.data.templates import V1C_FAMILIES, Gen, generate

REPO = Path(__file__).resolve().parents[1]
ARITH_NEW = sorted(set(Gen.ARITH_FAMILIES_V1C))
LINALG_NEW = ["linalg_ew2", "linalg_ewun2"]
OP_RE = re.compile(r"^\s{4}(?:%\S+ = )?([a-z]+\.[a-z_]+)", re.M)


def draws(family: str, n: int, seed: int = 11) -> list:
    """`n` programs of one new family (linalg_ew2 also yields linalg_ewun2; filter by name)."""
    g = Gen(seed, "v1c"); out = []
    method = "linalg_ew2" if family in LINALG_NEW else family
    while len(out) < n:
        p = getattr(g, method)()
        if p.family == family:
            out.append(p)
    return out


@pytest.fixture(scope="module")
def per_family() -> dict[str, list]:
    return {f: draws(f, 20) for f in ARITH_NEW + LINALG_NEW}


def test_family_sets_and_flag():
    assert set(ARITH_NEW) | set(LINALG_NEW) == set(V1C_FAMILIES)
    assert not (set(V1C_FAMILIES) & set(Gen.ARITH_FAMILIES))
    with pytest.raises(ValueError):
        Gen(0, "v2")
    old = generate(300, 60, seed=0)
    assert not any(r["family"] in V1C_FAMILIES for r in old)
    assert all(r["id"].startswith("tpl_0_") for r in old)
    new = generate(600, 200, seed=0, families="v1c")
    assert any(r["family"] in V1C_FAMILIES for r in new)
    assert all(r["id"].startswith("tpl_v1c_0_") for r in new)


def test_v1_pool_regenerates_bit_for_bit():
    """The v1 family set is untouched: every released train_v1 row is the candidate its id points to."""
    pool = REPO / "data/pools/train_v1.jsonl"
    if not pool.exists():
        pytest.skip("released v1 pool not present")
    rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
    cands = generate(18000, 6000, seed=0)
    for r in rows[::37]:
        c = cands[int(r["id"].rsplit("_", 1)[1])]
        assert (c["id"], c["nl"], c["mlir"], c["family"]) == (r["id"], r["nl"], r["mlir"], r["family"])


@pytest.mark.parametrize("family", ARITH_NEW + LINALG_NEW)
def test_family_golds_parse_and_fit_the_op_cap(per_family, family):
    for p in per_family[family]:
        m = p.mlir()
        assert p.family == family
        assert p.nl.split(" that ")[0] in {"Write a function", "Write an MLIR function", "Implement a function", "Define a function",
                                            "Create a function", f"Write a function `{p.name}`", f"Implement `{p.name}`, a function",
                                            "Write MLIR for a function"}, p.nl
        assert len(p.ops) <= 5, m  # generation grammar caps the body at 6 ops including `return`
        assert is_parse_valid(m), m


def test_new_families_cover_the_invented_concepts(per_family):
    """Each family emits the ops it exists to teach."""
    want = {"alloc_dealloc": {"memref.alloc", "memref.dealloc"}, "negate": {"arith.subi", "arith.subf"}, "clamp": {"arith.maxsi", "arith.minsi"},
            "cmp_const": {"arith.cmpi", "arith.cmpf"}, "const_op2": {"arith.shli", "arith.andi", "arith.remsi"},
            "cast_chain": {"arith.extsi", "arith.extf", "arith.sitofp", "arith.index_cast", "arith.fptosi"}, "bitcast": {"arith.bitcast"},
            "fminmax": {"arith.cmpf", "arith.select"}, "cond_select": {"arith.select"}, "mem_elem": {"memref.load", "memref.store"},
            "dim_cast": {"memref.dim", "arith.index_cast"}, "poly": {"arith.muli", "arith.mulf"},
            "linalg_ew2": {"linalg.add", "linalg.sub", "linalg.mul", "linalg.div"}, "linalg_ewun2": {"linalg.exp", "linalg.abs"}}
    for fam, ops in want.items():
        seen = {op for p in draws(fam, 60) for op in OP_RE.findall(p.mlir())}
        assert ops & seen, (fam, seen)


@pytest.mark.mlx
@pytest.mark.parametrize("family", ARITH_NEW + LINALG_NEW)
def test_family_golds_replay_under_generation_grammar(per_family, family):
    pytest.importorskip("mlx.core")
    from huggingface_hub import snapshot_download
    from mlx_lm.utils import load_tokenizer
    from sci.masks.llg import lark_grammar, llg_tokenizer, replay_allowed_sets
    hf = load_tokenizer(Path(snapshot_download("HuggingFaceTB/SmolLM2-135M-Instruct", allow_patterns=["*.json", "*.txt"])))
    tok = llg_tokenizer(hf); gram = lark_grammar("mlir_gen_c1c2")
    for p in per_family[family]:
        m = p.mlir()
        replay_allowed_sets(tok, gram, hf.encode(m, add_special_tokens=False))  # raises ValueError if any token is illegal


@pytest.mark.docker
@pytest.mark.parametrize("family", ARITH_NEW + LINALG_NEW)
def test_family_golds_verify_with_mlir_opt(per_family, family):
    from sci.eval.verify import verify
    for p in per_family[family][:5]:
        r = verify(p.mlir())
        assert r["returncode"] == 0, (family, r["stderr"], p.mlir())
