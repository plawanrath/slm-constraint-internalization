"""Prompting and generation: free, C1/C2-masked, and C3 rejection sampling.

Prompt format is the 3-shot chat prompt of the prior study, rendered through each
model's chat template so it ports across families. Constraint stacks:
  none     — free decoding
  c1       — token mask from `mlir_gen_c1.lark`
  c1_c2    — token mask from `mlir_gen_c1c2.lark`
  c1_c2_c3 — c1_c2 plus the post-hoc scope validator with up to 5 attempts
             (attempt 1 greedy, retries at temp 0.8, seeded per (seed, attempt))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from sci.constraints.scope import accept_or_reject
from sci.masks.llg import GrammarLogitsProcessor, lark_grammar, llg_tokenizer

SYSTEM = "Output only valid MLIR. Reuse parameter names exactly."

FEW_SHOT = {
    "arith+func": """Example 1:
Task: Write a function that adds two i32 values.
MLIR:
module {
  func.func @f(%a : i32 , %b : i32) -> i32 {
    %0 = arith.addi %a , %b : i32
    return %0 : i32
  }
}

Example 2:
Task: Write a function returning the i32 constant 42.
MLIR:
module {
  func.func @f() -> i32 {
    %0 = arith.constant 42 : i32
    return %0 : i32
  }
}

Example 3:
Task: Load element at index i from a memref.
MLIR:
module {
  func.func @f(%m : memref<?xf32> , %i : index) -> f32 {
    %0 = memref.load %m[%i] : memref<?xf32>
    return %0 : f32
  }
}
""",
    "linalg": """Example 1:
Task: Perform matmul of two 2-D f32 memrefs into an output memref.
MLIR:
module {
  func.func @mm(%A : memref<?x?xf32> , %B : memref<?x?xf32> , %C : memref<?x?xf32>) {
    linalg.matmul ins(%A , %B : memref<?x?xf32>, memref<?x?xf32>) outs(%C : memref<?x?xf32>)
    return
  }
}

Example 2:
Task: Fill a 1-D f32 memref with a given f32 value.
MLIR:
module {
  func.func @fill(%v : f32 , %m : memref<?xf32>) {
    linalg.fill ins(%v : f32) outs(%m : memref<?xf32>)
    return
  }
}

Example 3:
Task: Apply elementwise exp to a 1-D f32 memref.
MLIR:
module {
  func.func @ex(%x : memref<?xf32> , %y : memref<?xf32>) {
    linalg.exp ins(%x : memref<?xf32>) outs(%y : memref<?xf32>)
    return
  }
}
""",
}

GRAMMAR_FOR = {"c1": "mlir_gen_c1", "c1_c2": "mlir_gen_c1c2", "c1_c2_c3": "mlir_gen_c1c2"}
C3_MAX_ATTEMPTS = 5
RETRY_TEMP = 0.8


def user_message(nl: str, dialect: str) -> str:
    return f"{FEW_SHOT[dialect]}\n\nNow this task:\nTask: {nl}\nMLIR:"


def chatml_prompt(nl: str, dialect: str) -> str:
    """Literal ChatML rendering (byte-identical to the prior study's SmolLM2 prompt)."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user_message(nl, dialect)}"
            f"<|im_end|>\n<|im_start|>assistant\n")


def build_prompt(tokenizer, nl: str, dialect: str) -> str:
    """Render through the model's chat template; falls back to ChatML."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(nl, dialect)}]
    tpl = getattr(tokenizer, "chat_template", None)
    if not tpl:
        return chatml_prompt(nl, dialect)
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, **kwargs)


@dataclass
class GenOut:
    text: str
    tokens: list[int]
    prompt_tokens: int
    dt: float
    constraint: str
    attempts: int = 1
    scope_passed_per_try: list[bool] = field(default_factory=list)
    mask_errors: list[str] = field(default_factory=list)
    finished: bool = True  # False when max_tokens was hit


class Generator:
    """One loaded model + lazily compiled grammars."""

    def __init__(self, model_path: str, revision: str | None = None):
        self.model, self.tokenizer = mlx_load(model_path, revision=revision)
        self.llg_tok = llg_tokenizer(self.tokenizer)
        self._grammars: dict[str, str] = {}
        self.eos_ids = set(getattr(self.tokenizer, "eos_token_ids", None) or [self.tokenizer.eos_token_id])

    def grammar(self, name: str) -> str:
        if name not in self._grammars:
            self._grammars[name] = lark_grammar(name)
        return self._grammars[name]

    def encode(self, prompt: str) -> list[int]:
        add_bos = self.tokenizer.bos_token is None or not prompt.startswith(self.tokenizer.bos_token)
        return self.tokenizer.encode(prompt, add_special_tokens=add_bos)

    def _run(self, prompt_ids: list[int], constraint: str, max_tokens: int, temp: float) -> GenOut:
        t0 = time.perf_counter()
        proc = None
        if constraint != "none":
            proc = GrammarLogitsProcessor(self.llg_tok, self.grammar(GRAMMAR_FOR[constraint]), len(prompt_ids))
        sampler = make_sampler(temp=temp)
        out: list[int] = []
        finished = False
        for tok, _ in generate_step(mx.array(prompt_ids), self.model, max_tokens=max_tokens, sampler=sampler,
                                    logits_processors=[proc] if proc else None):
            tok = int(tok)
            if tok in self.eos_ids:
                finished = True; break
            out.append(tok)
            if proc is not None and proc.matcher.is_stopped():
                finished = True; break
        text = self.tokenizer.decode(out)
        return GenOut(text=text, tokens=out, prompt_tokens=len(prompt_ids), dt=time.perf_counter() - t0,
                      constraint=constraint, mask_errors=list(proc.errors) if proc else [], finished=finished)

    def generate(self, prompt: str, constraint: str = "none", max_tokens: int = 600, temp: float = 0.0,
                 seed: int = 0) -> GenOut:
        ids = self.encode(prompt)
        if constraint != "c1_c2_c3":
            mx.random.seed(seed)
            return self._run(ids, constraint, max_tokens, temp)
        # C3: greedy first attempt, then seeded temp-0.8 retries
        t0 = time.perf_counter()
        g = self._run(ids, "c1_c2", max_tokens, 0.0)
        ok, _ = accept_or_reject(g.text)
        passed = [ok]
        attempts = 1
        while not ok and attempts < C3_MAX_ATTEMPTS:
            attempts += 1
            mx.random.seed(seed * 1000 + attempts)
            g = self._run(ids, "c1_c2", max_tokens, RETRY_TEMP)
            ok, _ = accept_or_reject(g.text)
            passed.append(ok)
        g.constraint = "c1_c2_c3"; g.attempts = attempts; g.scope_passed_per_try = passed
        g.dt = time.perf_counter() - t0
        return g
