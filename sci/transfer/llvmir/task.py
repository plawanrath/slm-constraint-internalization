"""Prompt builder for the LLVM IR transfer target (3-shot, chat-template rendered).

Mirrors `sci.eval.generate.build_prompt`: a system line, a user message holding three
worked examples plus the task, rendered through the model's chat template with a
literal ChatML fallback. Generation itself goes through `LlvmGenerator` in
`sci.transfer.llvmir.generator`.
"""
from __future__ import annotations

SYSTEM = "Output only valid LLVM IR."
DIALECT = "llvmir"

FEW_SHOT = """Example 1:
Task: Write a function `f` that adds two i32 values and returns the sum.
LLVM IR:
define i32 @f(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

Example 2:
Task: Write a function `f` that returns the larger of two i64 values.
LLVM IR:
define i64 @f(i64 %a, i64 %b) {
entry:
  %cmp = icmp sgt i64 %a, %b
  %max = select i1 %cmp, i64 %a, i64 %b
  ret i64 %max
}

Example 3:
Task: Write a function `f` that returns x * 2.0 if the i32 flag is nonzero, otherwise x, for a float x.
LLVM IR:
define float @f(float %x, i32 %flag) {
entry:
  %nz = icmp ne i32 %flag, 0
  br i1 %nz, label %scale, label %done
scale:
  %dbl = fmul float %x, 2.0
  br label %done
done:
  %r = phi float [ %dbl, %scale ], [ %x, %entry ]
  ret float %r
}
"""


def user_message(nl: str) -> str:
    return f"{FEW_SHOT}\nNow this task:\nTask: {nl}\nLLVM IR:"


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
