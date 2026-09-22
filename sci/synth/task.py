"""Task adapter: everything the training driver needs to run on the synthetic language.

The MLIR driver hardcodes its grammar name, few-shot prompt builder, reward and C3 validator. This
module packages the synthetic equivalents behind one object with the intended `Task` protocol:

    build_prompt(tokenizer, nl) -> str     chat-templated 3-shot prompt
    grammar_name: str                      llguidance grammar for the C1+C2 mask
    grammar_lark(name) -> str              LARK text (the driver compiles it once per Generator)
    reward(text, gold, kind) -> (r, parts) sequence reward, parts has parse/scope/verify
    scope_ok(text) -> bool                 C3 verdict (for v3 accounting)
    gold_key: str                          pool column holding the gold program

`MLIRTask` wraps the existing MLIR pieces with the same surface so the driver can switch by name.
"""
from __future__ import annotations

from sci.synth import checker
from sci.synth.grammar import grammar_text

SYSTEM = "Output only a valid prog. Use the statement syntax exactly."

FEW_SHOT = """Example 1:
Task: Write a prog that keeps 2 values live at once, starting with 2 constants (2 int[4]). The constant at position 1 is a int[4]; leave it untouched for exactly 1 statements, then combine it with addi. Ops used in total: addi x1, const x2. Return the last value, a int[4].
Prog:
prog {
  %v0 = const 3 : int[4]
  %v1 = const -7 : int[4]
  %v2 = addi %v0 , %v1 : int[4]
  ret %v2 : int[4]
}

Example 2:
Task: Emit a prog that keeps 2 values live at once, starting with 2 constants (1 flt[2x3], 1 int[8]). The constant at position 1 is a flt[2x3]; leave it untouched for exactly 2 statements, then combine it with mulf. Ops used in total: const x3, mulf x1. Return the last value, a flt[2x3].
Prog:
prog {
  %v0 = const 1.5 : flt[2x3]
  %v1 = const 4 : int[8]
  %v2 = const 0.5 : flt[2x3]
  %v3 = mulf %v2 , %v0 : flt[2x3]
  ret %v3 : flt[2x3]
}

Example 3:
Task: Produce a prog that keeps 2 values live at once, starting with 2 constants (2 int[3x2]). The constant at position 2 is a int[3x2]; leave it untouched for exactly 2 statements, then combine it with subi. The program has 1 nested block whose values are dropped afterwards. Ops used in total: const x2, muli x2, subi x1. Return the last value, a int[3x2].
Prog:
prog {
  %v0 = const 12 : int[3x2]
  %v1 = const -3 : int[3x2]
  {
    %v2 = muli %v0 , %v0 : int[3x2]
    %v3 = muli %v2 , %v0 : int[3x2]
  }
  %v4 = subi %v1 , %v0 : int[3x2]
  ret %v4 : int[3x2]
}
"""


def user_message(nl: str) -> str:
    return f"{FEW_SHOT}\nNow this task:\nTask: {nl}\nProg:"


def chatml_prompt(nl: str) -> str:
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user_message(nl)}"
            f"<|im_end|>\n<|im_start|>assistant\n")


class SynthTask:
    """The synthetic-language task (see module docstring for the protocol)."""
    name = "synth"
    grammar_name = "ssa_gen"
    c1_grammar_name = "ssa_gen_c1"
    gold_key = "gold"

    @staticmethod
    def grammar_lark(name: str = "ssa_gen") -> str:
        return grammar_text(name)

    @staticmethod
    def grammar(name: str = "ssa_gen") -> str:
        """Compiled llguidance spec (imports llguidance lazily)."""
        from llguidance import LLMatcher
        return LLMatcher.grammar_from_lark(grammar_text(name))

    @staticmethod
    def build_prompt(tokenizer, nl: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(nl)}]
        if not getattr(tokenizer, "chat_template", None):
            return chatml_prompt(nl)
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(msgs, **kwargs)

    @staticmethod
    def reward(text: str, gold: str | None = None, kind: str = "task") -> tuple[float, dict]:
        return checker.reward_fn(text, gold, kind)

    @staticmethod
    def scope_ok(text: str) -> bool:
        return checker.scope_ok(text)

    @staticmethod
    def accept_or_reject(text: str):
        return checker.accept_or_reject(text)

    @staticmethod
    def check(text: str) -> dict:
        return checker.check(text).as_dict()


class MLIRTask:
    """The existing MLIR task behind the same protocol (imports the MLIR stack lazily)."""
    name = "mlir"
    grammar_name = "mlir_gen_c1c2"
    c1_grammar_name = "mlir_gen_c1"
    gold_key = "mlir"

    @staticmethod
    def grammar_lark(name: str = "mlir_gen_c1c2") -> str:
        from sci.constraints.parser import generation_grammar_path
        return generation_grammar_path(name).read_text()

    @staticmethod
    def build_prompt(tokenizer, nl: str, dialect: str = "arith+func") -> str:
        from sci.eval.generate import build_prompt
        return build_prompt(tokenizer, nl, dialect)

    @staticmethod
    def reward(text: str, gold: str | None = None, kind: str = "task") -> tuple[float, dict]:
        from sci.train.spine import reward_fn
        return reward_fn(text, gold, kind)

    @staticmethod
    def scope_ok(text: str) -> bool:
        from sci.constraints.scope import accept_or_reject
        return accept_or_reject(text)[0]


def _llvm_task():
    from sci.transfer.llvmir.train_task import LlvmTask  # second training target (textual LLVM IR)
    return LlvmTask()


TASKS = {"synth": SynthTask, "mlir": MLIRTask, "llvm": _llvm_task}


def get_task(name: str):
    return TASKS[name]()


def prompt_for_row(task, tokenizer, row: dict) -> str:
    """Prompt for one pool row (passes `dialect` through only for tasks whose rows carry one)."""
    if task.name == "mlir":
        return task.build_prompt(tokenizer, row["nl"], row["dialect"])
    return task.build_prompt(tokenizer, row["nl"])


def gold_for_row(task, row: dict) -> str:
    return row[task.gold_key]
