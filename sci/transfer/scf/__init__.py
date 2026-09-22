"""New-MLIR-dialect transfer target: structured control flow (scf).

`scf.for` / `scf.if` / `scf.yield` with region block arguments and iter_args on
top of the arith + memref subset. Generation grammars (C1 / C1+C2), a
region-aware SSA scope validator (C3), the tblgen-mined lattice, a
gold-differential functional wrapper for the existing `mlir-cpu-runner` harness,
and the prompt builder / generator glue for the Scf-Spec-60 test set.
"""
