"""Generation grammars for the synthetic SSA language (llguidance-compatible LARK).

Two bundled grammars, both rendered by `render_grammar` and stored under `grammars/`:
  ssa_gen     — C1 syntax + C2 type domains (int ops take `int[...]`, flt ops `flt[...]`,
                integer literals pair with int types, decimal literals with flt types)
  ssa_gen_c1  — C1 only: any op / literal with any type

Style follows the MLIR generation grammars: explicit `WS` productions (no `%ignore`), bounded
identifiers and whitespace, and statement lists capped by explicit rule chains (llguidance has no
`~n..m` repetition), one chain per nesting level so block depth is bounded too.

Surface form of a program (one statement per line, two-space indentation per level):

    prog {
      %v0 = const 3 : int[4]
      %v1 = const 1.5 : flt[2x3]
      {
        %v2 = addi %v0 , %v0 : int[4]
      }
      %v3 = subi %v0 , %v0 : int[4]
      ret %v3 : int[4]
    }
"""
from __future__ import annotations

from pathlib import Path

GRAMMAR_DIR = Path(__file__).resolve().parent / "grammars"
MAX_STMTS = 128      # statements per block (each nesting level has its own chain)
MAX_DEPTH = 2        # blocks may nest this deep below the program body
GRAMMAR_NAMES = ("ssa_gen", "ssa_gen_c1")

INT_OPS = ("addi", "subi", "muli")
FLT_OPS = ("addf", "subf", "mulf")
DOMAINS = ("int", "flt")

_HEADER = """// Generation grammar for the synthetic SSA language ({kind}).
// Rendered by sci.synth.grammar.render_grammar; do not edit by hand.
// Explicit WS productions, no %ignore, bounded identifiers/whitespace, statement lists capped
// at {max_stmts} per block by rule chains, block nesting capped at depth {max_depth}.

start: "prog" WS "{{" WS s0_0 WS ret_stmt WS "}}" WS?

ret_stmt: "ret" WS SSA WS ":" WS type
"""

_TYPED_OPS = """
op_stmt: const_int | const_flt | bin_int | bin_flt

const_int: SSA WS "=" WS "const" WS INT_LIT WS ":" WS int_type
const_flt: SSA WS "=" WS "const" WS FLT_LIT WS ":" WS flt_type
bin_int:   SSA WS "=" WS INT_BIN WS SSA WS "," WS SSA WS ":" WS int_type
bin_flt:   SSA WS "=" WS FLT_BIN WS SSA WS "," WS SSA WS ":" WS flt_type
"""

_UNTYPED_OPS = """
op_stmt: const_stmt | bin_stmt

const_stmt: SSA WS "=" WS "const" WS lit WS ":" WS type
bin_stmt:   SSA WS "=" WS BIN WS SSA WS "," WS SSA WS ":" WS type
lit: INT_LIT | FLT_LIT
BIN: INT_BIN | FLT_BIN
"""

_TAIL = """
type: int_type | flt_type
int_type: "int" shape
flt_type: "flt" shape
shape: "[" DIM "]" | "[" DIM "x" DIM "]"
DIM: /[1-9][0-9]?/

INT_BIN: {int_bin}
FLT_BIN: {flt_bin}

INT_LIT: /-?[0-9]{{1,4}}/
FLT_LIT: /-?[0-9]{{1,4}}\\.[0-9]{{1,3}}/

SSA: /%[a-z][a-z0-9]{{0,7}}/
WS: /[ \\t\\n]{{1,8}}/
"""


def _chain(depth: int, max_stmts: int, max_depth: int) -> str:
    """Rule chain `s{depth}_i` for one nesting level: 1..max_stmts statements."""
    lines = []
    for i in range(max_stmts):
        if i < max_stmts - 1:
            lines.append(f"s{depth}_{i}: stmt{depth} | stmt{depth} WS s{depth}_{i + 1}")
        else:
            lines.append(f"s{depth}_{i}: stmt{depth}")
    if depth < max_depth:
        lines.append(f"stmt{depth}: op_stmt | block{depth}")
        lines.append(f'block{depth}: "{{" WS s{depth + 1}_0 WS "}}"')
    else:
        lines.append(f"stmt{depth}: op_stmt")
    return "\n".join(lines) + "\n"


def render_grammar(typed: bool = True, max_stmts: int = MAX_STMTS, max_depth: int = MAX_DEPTH) -> str:
    """LARK text of the C1+C2 (`typed=True`) or C1-only grammar."""
    out = [_HEADER.format(kind="C1 syntax + C2 type domains" if typed else "C1 syntax only",
                          max_stmts=max_stmts, max_depth=max_depth)]
    out.append(_TYPED_OPS if typed else _UNTYPED_OPS)
    out.append("\n")
    for d in range(max_depth + 1):
        out.append(_chain(d, max_stmts, max_depth))
    out.append(_TAIL.format(int_bin=" | ".join(f'"{o}"' for o in INT_OPS),
                            flt_bin=" | ".join(f'"{o}"' for o in FLT_OPS)))
    return "".join(out)


def grammar_path(name: str = "ssa_gen") -> Path:
    if name not in GRAMMAR_NAMES:
        raise KeyError(f"unknown synthetic grammar {name!r}; choose from {GRAMMAR_NAMES}")
    return GRAMMAR_DIR / f"{name}.lark"


def grammar_text(name: str = "ssa_gen") -> str:
    """LARK text of a bundled synthetic grammar (`ssa_gen` or `ssa_gen_c1`)."""
    return grammar_path(name).read_text()


def compile_grammar(name: str = "ssa_gen") -> str:
    """llguidance grammar spec for `LLMatcher` (same object `sci.masks.llg.lark_grammar` returns)."""
    from llguidance import LLMatcher
    return LLMatcher.grammar_from_lark(grammar_text(name))


def write_grammars(directory: Path = GRAMMAR_DIR) -> None:
    """(Re)write both bundled grammar files."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ssa_gen.lark").write_text(render_grammar(typed=True))
    (directory / "ssa_gen_c1.lark").write_text(render_grammar(typed=False))


if __name__ == "__main__":
    write_grammars()
