"""Prompt builder for the scf transfer target (3-shot, chat-template rendered).

Mirrors `sci.eval.generate.build_prompt`: the same system line, a user message with
three worked examples plus the task, rendered through the model's chat template
with a literal ChatML fallback. The examples show the three scf forms the grammar
emits (for + iter_args, if with a result, result-less if inside a loop).
`ScfGenerator` (the model-side counterpart) lives in `sci.transfer.scf.generator`
and is re-exported lazily so importing this module needs no MLX.
"""
from __future__ import annotations

# Same system line as `sci.eval.generate.SYSTEM` (not imported: that module loads MLX).
SYSTEM = "Output only valid MLIR. Reuse parameter names exactly."
DIALECT = "scf"

FEW_SHOT = """Example 1:
Task: Write a function `sum_to` that takes an index n and returns the i32 sum of the indices 0 .. n-1 using scf.for with an iter_args accumulator.
MLIR:
module {
  func.func @sum_to(%n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (i32) {
      %ii = arith.index_cast %i : index to i32
      %s = arith.addi %acc, %ii : i32
      scf.yield %s : i32
    }
    return %r : i32
  }
}

Example 2:
Task: Write a function `fmax` that takes two f32 values and returns the larger one using scf.if with an f32 result.
MLIR:
module {
  func.func @fmax(%a: f32, %b: f32) -> f32 {
    %c = arith.cmpf ogt, %a, %b : f32
    %r = scf.if %c -> (f32) {
      scf.yield %a : f32
    } else {
      scf.yield %b : f32
    }
    return %r : f32
  }
}

Example 3:
Task: Write a function `zero_neg` that takes a 1-D f32 memref with dynamic shape and stores 0.0 over every negative element, using scf.for over memref.dim and a result-less scf.if.
MLIR:
module {
  func.func @zero_neg(%a: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %n = memref.dim %a, %c0 : memref<?xf32>
    scf.for %i = %c0 to %n step %c1 {
      %v = memref.load %a[%i] : memref<?xf32>
      %neg = arith.cmpf olt, %v, %zero : f32
      scf.if %neg {
        memref.store %zero, %a[%i] : memref<?xf32>
      }
    }
    return
  }
}
"""


def user_message(nl: str) -> str:
    return f"{FEW_SHOT}\nNow this task:\nTask: {nl}\nMLIR:"


def chatml_prompt(nl: str) -> str:
    """Literal ChatML rendering (fallback when a tokenizer has no chat template)."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user_message(nl)}"
            f"<|im_end|>\n<|im_start|>assistant\n")


def build_prompt(tokenizer, nl: str, dialect: str = DIALECT) -> str:
    """Render through the model's chat template; `dialect` is accepted for API symmetry."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(nl)}]
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return chatml_prompt(nl)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


def __getattr__(name: str):
    if name == "ScfGenerator":
        from sci.transfer.scf.generator import ScfGenerator
        return ScfGenerator
    raise AttributeError(name)
