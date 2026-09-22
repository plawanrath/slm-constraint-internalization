"""Portability ladder: trained checkpoints (and optional untrained bases) on the held-out
LLVM-Spec-60 target under free / C1 / C1+C2 / C1+C2+C3, plus per-layer residuals and a
memorization check against the training golds.

  python scripts/w12_portability.py --tags smollm2-360m-instruct_ssd_s0 \
      [--base HuggingFaceTB/SmolLM2-360M-Instruct] [--stablehlo] [--limit N]

Per tag the checkpoint is `models/ckpt/<tag>/best` (its `run_config.json` names the base
model, revision and method). `--stablehlo` adds the StableHLO Spec-30 / Held-Out-200 pools
through `sci.eval.verify_stablehlo`. Writes `results/w12_portability/ladder.jsonl` (resumable:
rows already present are skipped) and `summary.json`.

Per-layer violation flags on each generation (LLVM): v1 = grammar-invalid, v2 = llvm-as
type-domain diagnostic, v3 = scope-validator failure or llvm-as dominance/undefined-value
diagnostic, v4 = functional signature mismatch. Residual_k = v_k(free) - v_k(stack k) with
paired bootstrap CIs (`sci.eval.stats.paired_bootstrap_diff`). Memorization = max over
training golds of the fraction of a generation's 8-grams that the gold contains.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sci.eval.stats import bootstrap_ci, paired_bootstrap_diff  # noqa: E402

OUT = REPO / "results" / "w12_portability"
CONSTRAINTS = ("none", "c1", "c1_c2", "c1_c2_c3")
LLVM_POOL = "llvm_spec_60"
STABLEHLO_POOLS = {"stablehlo_spec_30": "data/benchmarks/stablehlo_spec_30.jsonl",
                   "stablehlo_held_out_200": "data/benchmarks/stablehlo_held_out_200.jsonl"}
TRAIN_POOL = REPO / "data" / "pools" / "train_v1.jsonl"
STACK_FOR = {"v1": "c1", "v2": "c1_c2", "v3": "c1_c2_c3", "v4": "c1_c2_c3"}

STABLEHLO_FEW_SHOT = """Example 1:
Task: Add two 1-D f32 tensors of 16 elements.
MLIR:
module {
  func.func @f(%a : tensor<16xf32> , %b : tensor<16xf32>) -> tensor<16xf32> {
    %0 = stablehlo.add %a , %b : tensor<16xf32>
    return %0 : tensor<16xf32>
  }
}

Example 2:
Task: Apply elementwise absolute value to a 2-D f32 tensor of shape 4x8.
MLIR:
module {
  func.func @f(%a : tensor<4x8xf32>) -> tensor<4x8xf32> {
    %0 = stablehlo.abs %a : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}

Example 3:
Task: Transpose a 2-D f32 tensor of shape 4x8 into 8x4.
MLIR:
module {
  func.func @f(%a : tensor<4x8xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.transpose %a, dims = [1, 0] : (tensor<4x8xf32>) -> tensor<8x4xf32>
    return %0 : tensor<8x4xf32>
  }
}
"""


# ---------------- memorization ----------------

_TOK_RE = re.compile(r"\w+|[^\w\s]")


def ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    toks = _TOK_RE.findall(text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


class MemorizationIndex:
    """8-gram inverted index over the training golds."""

    def __init__(self, golds: list[str], n: int = 8):
        self.n = n
        self.index: dict[tuple, set[int]] = {}
        self.norm = {re.sub(r"\s+", " ", g).strip() for g in golds}
        for gid, g in enumerate(golds):
            for ng in ngrams(g, n):
                self.index.setdefault(ng, set()).add(gid)

    def max_overlap(self, text: str) -> dict:
        grams = ngrams(text, self.n)
        if not grams:
            return {"max_8gram_overlap": 0.0, "n_8grams": 0, "exact": False}
        hits: Counter = Counter()
        for ng in grams:
            for gid in self.index.get(ng, ()):
                hits[gid] += 1
        best = max(hits.values()) if hits else 0
        return {"max_8gram_overlap": round(best / len(grams), 4), "n_8grams": len(grams),
                "exact": re.sub(r"\s+", " ", text).strip() in self.norm}


def load_train_golds() -> list[str]:
    return [json.loads(l)["mlir"] for l in TRAIN_POOL.read_text().splitlines() if l.strip()]


# ---------------- target-specific scoring ----------------

class LlvmTarget:
    pool = LLVM_POOL

    def __init__(self):
        from sci.transfer.llvmir import functional as fn
        from sci.transfer.llvmir.grammar import is_parse_valid
        from sci.transfer.llvmir.scope import accept_or_reject
        from sci.transfer.llvmir.task import build_prompt
        from sci.transfer.llvmir.verify import categorize, verify
        self.fn, self.parse, self.scope, self.build_prompt = fn, is_parse_valid, accept_or_reject, build_prompt
        self.verify, self.categorize = verify, categorize
        self.rows = [json.loads(l) for l in (REPO / "data/benchmarks/llvm_spec_60.jsonl").read_text().splitlines() if l.strip()]
        self.refs = {r["prompt_id"]: r for r in
                     (json.loads(l) for l in (REPO / "data/functional/llvm_references.jsonl").read_text().splitlines() if l.strip())}
        self.gate_path = OUT / "gold_gate_llvm.json"
        self.gates: dict = json.loads(self.gate_path.read_text()) if self.gate_path.exists() else {}

    def make_generator(self, repo: str, revision: str | None):
        from sci.transfer.llvmir.generator import LlvmGenerator
        return LlvmGenerator(repo, revision=revision)

    def gate(self, pid: int) -> dict:
        key = str(pid)
        if key not in self.gates:
            self.gates[key] = self.fn.gold_gate(self.rows[pid]["gold"], self.refs[pid])
            self.gate_path.parent.mkdir(parents=True, exist_ok=True)
            self.gate_path.write_text(json.dumps(self.gates, indent=1))
        return self.gates[key]

    def score(self, pid: int, text: str) -> dict:
        pv = self.parse(text, "llvm_gen_c1")
        v = self.verify(text)
        vv = v["returncode"] == 0
        cat = "passed" if vv else self.categorize(v["stderr"])
        scope_ok, _ = self.scope(text)
        gate = self.gate(pid)
        if not gate["ok"]:
            fstatus, fmatch = "gold_gate_fail", False
        elif not vv:
            fstatus, fmatch = "not_verify_valid", False
        else:
            d = self.fn.differential(text, self.rows[pid]["gold"], self.refs[pid], gold_outputs=gate["outputs"])
            fstatus, fmatch = d["status"], d["status"] == "match"
        return {"parse_valid": pv, "verify_valid": vv, "verify_cat": cat, "scope_ok": scope_ok,
                "functional_status": fstatus, "functional_match": fmatch,
                "v1": not pv, "v2": pv and cat == "type", "v3": pv and (not scope_ok or cat == "scope"),
                "v4": fstatus == "signature_mismatch"}


class StablehloTarget:
    def __init__(self, pool: str):
        from sci.constraints.parser import is_parse_valid
        from sci.constraints.scope import accept_or_reject
        from sci.eval.error_categories import categorize
        from sci.eval.residual import SHAPE_RE
        from sci.eval.verify_stablehlo import verify_stablehlo
        self.pool = pool
        self.parse, self.scope, self.categorize, self.shape_re = is_parse_valid, accept_or_reject, categorize, SHAPE_RE
        self.verify = verify_stablehlo
        self.rows = [json.loads(l) for l in (REPO / STABLEHLO_POOLS[pool]).read_text().splitlines() if l.strip()]

    def make_generator(self, repo: str, revision: str | None):
        from sci.eval.generate import Generator
        from sci.masks.llg import lark_grammar

        class StablehloGenerator(Generator):
            def grammar(self, name: str) -> str:
                if "mlir_gen_stablehlo" not in self._grammars:
                    self._grammars["mlir_gen_stablehlo"] = lark_grammar("mlir_gen_stablehlo")
                return self._grammars["mlir_gen_stablehlo"]

        return StablehloGenerator(repo, revision=revision)

    def build_prompt(self, tokenizer, nl: str) -> str:
        from sci.eval import generate as g
        saved = dict(g.FEW_SHOT)
        g.FEW_SHOT["stablehlo"] = STABLEHLO_FEW_SHOT
        try:
            return g.build_prompt(tokenizer, nl, "stablehlo")
        finally:
            g.FEW_SHOT.clear(); g.FEW_SHOT.update(saved)

    def score(self, pid: int, text: str) -> dict:
        pv = self.parse(text)
        v = self.verify(text)
        vv = v["returncode"] == 0
        cat = "passed" if vv else self.categorize(v["stderr"])
        scope_ok, _ = self.scope(text)
        shape = (not vv) and bool(self.shape_re.search(v["stderr"] or "")) and cat not in ("type", "type_ssa")
        return {"parse_valid": pv, "verify_valid": vv, "verify_cat": cat, "scope_ok": scope_ok,
                "functional_status": "n/a", "functional_match": False,
                "v1": not pv, "v2": pv and cat == "type", "v3": pv and (not scope_ok or cat == "type_ssa"),
                "v4": pv and shape}


# ---------------- ladder ----------------

def done_keys(path: Path) -> set[tuple]:
    keys = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                keys.add((r["model"], r["pool"], r["constraint"], r["prompt_id"]))
    return keys


def resolve_model(tag: str, base: str | None) -> dict:
    """{repo, revision, method, ckpt_dir, tag} for a checkpoint tag or an untrained base repo."""
    if base is not None:
        rev = None
        for line in (REPO / "scripts/env/models.lock.txt").read_text().splitlines():
            if line.startswith(base + "@"):
                rev = line.split("@", 1)[1].strip()
        return {"repo": base, "revision": rev, "method": "none", "ckpt_dir": None,
                "tag": base.split("/")[-1].lower() + "_base"}
    ckpt = REPO / "models" / "ckpt" / tag / "best"
    cfg = json.loads((ckpt / "run_config.json").read_text())
    return {"repo": cfg["model"], "revision": cfg.get("revision"), "method": cfg["method"], "ckpt_dir": ckpt, "tag": tag}


def run_cells(spec: dict, targets: list, constraints: tuple[str, ...], out_jsonl: Path, memo: MemorizationIndex,
              max_tokens: int, seed: int, limit: int | None) -> None:
    done = done_keys(out_jsonl)
    todo = [(t, c, r["prompt_id"] if "prompt_id" in r else i)
            for t in targets for c in constraints for i, r in enumerate(t.rows[:limit] if limit else t.rows)]
    if all((spec["tag"], t.pool, c, pid) in done for t, c, pid in todo):
        print(f"[w12] {spec['tag']}: all cells present, skipping", file=sys.stderr)
        return
    gens: dict[str, object] = {}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("a") as fout:
        for target in targets:
            rows = target.rows[:limit] if limit else target.rows
            for constraint in constraints:
                n_done = n_vv = n_fm = 0; t_cell = time.perf_counter()
                for i, r in enumerate(rows):
                    pid = r["prompt_id"] if "prompt_id" in r else i
                    key = (spec["tag"], target.pool, constraint, pid)
                    if key in done:
                        continue
                    gkey = type(target).__name__
                    if gkey not in gens:
                        gen = target.make_generator(spec["repo"], spec["revision"])
                        if spec["ckpt_dir"] is not None:
                            from sci.train.run import load_ckpt
                            load_ckpt(gen, spec["ckpt_dir"])
                        gens[gkey] = gen
                    gen = gens[gkey]
                    row = {"model": spec["tag"], "method": spec["method"],
                           "ckpt": str(spec["ckpt_dir"]) if spec["ckpt_dir"] else "base", "train_seed": None,
                           "constraint": constraint, "pool": target.pool, "seed": seed, "prompt_id": pid, "nl": r["nl"]}
                    try:
                        prompt = target.build_prompt(gen.tokenizer, r["nl"])
                        g = gen.generate(prompt, constraint=constraint, max_tokens=max_tokens, seed=seed)
                        row.update({"generated": g.text, "attempts": g.attempts,
                                    "scope_passed_per_try": g.scope_passed_per_try, "finished": g.finished,
                                    "n_tokens": len(g.tokens), "dt": round(g.dt, 3)})
                        row.update(target.score(pid, g.text))
                        row["memorization"] = memo.max_overlap(g.text)
                    except Exception as e:  # noqa: BLE001
                        row.update({"generated": "", "attempts": 0, "scope_passed_per_try": [], "finished": False,
                                    "n_tokens": 0, "dt": 0.0, "parse_valid": False, "verify_valid": False,
                                    "verify_cat": "error", "scope_ok": False, "functional_status": "error",
                                    "functional_match": False, "v1": True, "v2": False, "v3": False, "v4": False,
                                    "memorization": {"max_8gram_overlap": 0.0, "n_8grams": 0, "exact": False},
                                    "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    fout.write(json.dumps(row) + "\n"); fout.flush()
                    n_done += 1; n_vv += row["verify_valid"]; n_fm += row["functional_match"]
                    if n_done % 20 == 0:
                        print(f"  [{spec['tag']}|{target.pool}|{constraint}] {n_done}/{len(rows)} verify={n_vv} "
                              f"functional={n_fm} {(time.perf_counter() - t_cell) / n_done:.2f}s/gen", file=sys.stderr)
                if n_done:
                    print(f"[w12] {spec['tag']}|{target.pool}|{constraint}: verify {n_vv}/{n_done} functional {n_fm}/{n_done} "
                          f"in {(time.perf_counter() - t_cell) / 60:.1f} min", file=sys.stderr)


# ---------------- summary ----------------

def summarize(jsonl: Path, out_json: Path) -> dict:
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    cells: dict[tuple, dict[int, dict]] = {}
    for r in rows:
        cells.setdefault((r["model"], r["pool"], r["constraint"]), {})[r["prompt_id"]] = r
    summary: dict = {}
    for (model, pool, constraint), byid in sorted(cells.items()):
        ids = sorted(byid)
        vv = bootstrap_ci([int(byid[i]["verify_valid"]) for i in ids])
        fm = bootstrap_ci([int(byid[i]["functional_match"]) for i in ids])
        mem = [byid[i].get("memorization", {}).get("max_8gram_overlap", 0.0) for i in ids]
        summary.setdefault(model, {}).setdefault(pool, {})[constraint] = {
            "n": len(ids),
            "parse_rate": round(sum(byid[i]["parse_valid"] for i in ids) / len(ids), 4),
            "verify_rate": round(vv.point, 4), "verify_ci95": [round(vv.ci_low, 4), round(vv.ci_high, 4)],
            "scope_rate": round(sum(byid[i]["scope_ok"] for i in ids) / len(ids), 4),
            "functional_rate": round(fm.point, 4), "functional_ci95": [round(fm.ci_low, 4), round(fm.ci_high, 4)],
            "functional_status_counts": dict(Counter(byid[i]["functional_status"] for i in ids)),
            "violation_rates": {k: round(sum(bool(byid[i][k]) for i in ids) / len(ids), 4) for k in ("v1", "v2", "v3", "v4")},
            "mean_attempts": round(sum(byid[i]["attempts"] for i in ids) / len(ids), 3),
            "unfinished": sum(1 for i in ids if not byid[i].get("finished", True)),
            "memorization": {"mean_max_8gram_overlap": round(sum(mem) / len(mem), 4), "max": round(max(mem), 4),
                             "frac_ge_0.5": round(sum(m >= 0.5 for m in mem) / len(mem), 4),
                             "n_exact": sum(1 for i in ids if byid[i].get("memorization", {}).get("exact"))},
        }
    for model in summary:
        for pool in summary[model]:
            residual, deltas = {}, {}
            base = cells.get((model, pool, "none"))
            for k, sc in STACK_FOR.items():
                stack = cells.get((model, pool, sc))
                if base and stack:
                    ids = sorted(set(base) & set(stack))
                    if ids:
                        d = paired_bootstrap_diff([int(bool(base[i][k])) for i in ids], [int(bool(stack[i][k])) for i in ids])
                        residual[k] = {"vs": sc, "pp": round(100 * d.point, 1),
                                       "ci95_pp": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)], "n": d.n}
            for metric in ("verify_valid", "functional_match"):
                for a, b in (("c1", "none"), ("c1_c2", "c1"), ("c1_c2_c3", "c1_c2"), ("c1_c2_c3", "none")):
                    ka, kb = cells.get((model, pool, a)), cells.get((model, pool, b))
                    if ka and kb:
                        ids = sorted(set(ka) & set(kb))
                        if ids:
                            d = paired_bootstrap_diff([int(ka[i][metric]) for i in ids], [int(kb[i][metric]) for i in ids])
                            deltas.setdefault(metric, {})[f"{a}-{b}"] = {
                                "pp": round(100 * d.point, 1), "ci95_pp": [round(100 * d.ci_low, 1), round(100 * d.ci_high, 1)],
                                "p_one_sided": round(d.p_value, 4), "n_pairs": d.n}
            summary[model][pool]["residual_pp"] = residual
            summary[model][pool]["paired_deltas"] = deltas
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="", help="comma-separated checkpoint tags under models/ckpt/<tag>/best")
    ap.add_argument("--base", action="append", default=[], help="untrained HF repo to evaluate as a base rung (repeatable)")
    ap.add_argument("--stablehlo", action="store_true", help="also run the StableHLO Spec-30 and Held-Out-200 pools")
    ap.add_argument("--constraints", default=",".join(CONSTRAINTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ladder = OUT / "ladder.jsonl"
    if not args.summary_only:
        targets: list = [LlvmTarget()]
        if args.stablehlo:
            targets += [StablehloTarget(p) for p in STABLEHLO_POOLS]
        memo = MemorizationIndex(load_train_golds())
        specs = [resolve_model(t, None) for t in args.tags.split(",") if t] + [resolve_model("", b) for b in args.base]
        if not specs:
            ap.error("give --tags and/or --base")
        for spec in specs:
            run_cells(spec, targets, tuple(args.constraints.split(",")), ladder, memo, args.max_tokens, args.seed, args.limit)
    if ladder.exists():
        s = summarize(ladder, OUT / "summary.json")
        print(json.dumps(s, indent=1)[:4000])


if __name__ == "__main__":
    main()
