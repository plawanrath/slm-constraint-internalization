"""`Generator` subclass that resolves the LLVM grammars and the LLVM scope validator.

`sci.eval.generate.Generator._run` looks up `GRAMMAR_FOR[constraint]` (an MLIR grammar
name) and calls `self.grammar(name)`; overriding `grammar` to map those names onto the
bundled LLVM grammars makes the whole free / C1 / C1+C2 path reusable unchanged. The
C1+C2+C3 rejection loop is re-implemented here so it uses the LLVM validator.
"""
from __future__ import annotations

import time

import mlx.core as mx

from sci.eval.generate import C3_MAX_ATTEMPTS, RETRY_TEMP, GenOut, Generator
from sci.transfer.llvmir.grammar import llg_grammar
from sci.transfer.llvmir.scope import accept_or_reject

# MLIR grammar name (as `Generator._run` requests it) -> LLVM grammar name.
_REMAP = {"mlir_gen_c1": "llvm_gen_c1", "mlir_gen_c1c2": "llvm_gen_c1c2"}


class LlvmGenerator(Generator):
    """Same model loading / decoding as `Generator`, with LLVM grammars and C3."""

    def grammar(self, name: str) -> str:
        name = _REMAP.get(name, name)
        if name not in self._grammars:
            self._grammars[name] = llg_grammar(name)
        return self._grammars[name]

    def generate(self, prompt: str, constraint: str = "none", max_tokens: int = 600, temp: float = 0.0,
                 seed: int = 0) -> GenOut:
        if constraint != "c1_c2_c3":
            return super().generate(prompt, constraint=constraint, max_tokens=max_tokens, temp=temp, seed=seed)
        ids = self.encode(prompt)
        t0 = time.perf_counter()
        mx.random.seed(seed)
        g = self._run(ids, "c1_c2", max_tokens, 0.0)
        ok, _ = accept_or_reject(g.text)
        passed, attempts = [ok], 1
        while not ok and attempts < C3_MAX_ATTEMPTS:
            attempts += 1
            mx.random.seed(seed * 1000 + attempts)
            g = self._run(ids, "c1_c2", max_tokens, RETRY_TEMP)
            ok, _ = accept_or_reject(g.text)
            passed.append(ok)
        g.constraint = "c1_c2_c3"; g.attempts = attempts; g.scope_passed_per_try = passed
        g.dt = time.perf_counter() - t0
        return g
