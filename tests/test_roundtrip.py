"""Grammar acceptance tests.

v0 goal: parse hand-crafted arith+func+memref snippets without error.
Day-2 goal: extend this to a 1k random sample from L3 and require 95%+ accept.
"""
from __future__ import annotations

import pytest

from sci.constraints.parser import is_parse_valid


ARITH_SAMPLES = [
    """module {
  func.func @f(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    return %0 : i32
  }
}""",
    """module {
  func.func @g() -> f32 {
    %c = arith.constant 3.14 : f32
    return %c : f32
  }
}""",
    """module {
  func.func @cmp(%a: i64, %b: i64) -> i1 {
    %0 = arith.cmpi slt, %a, %b : i64
    return %0 : i1
  }
}""",
]

MEMREF_SAMPLES = [
    """module {
  func.func @m() {
    %buf = memref.alloc() : memref<128xf32>
    memref.dealloc %buf : memref<128xf32>
    return
  }
}""",
    """module {
  func.func @l(%buf: memref<?xf32>, %i: index) -> f32 {
    %v = memref.load %buf[%i] : memref<?xf32>
    return %v : f32
  }
}""",
]


@pytest.mark.parametrize("src", ARITH_SAMPLES)
def test_arith_roundtrip(src: str) -> None:
    assert is_parse_valid(src), f"arith sample rejected: {src!r}"


@pytest.mark.parametrize("src", MEMREF_SAMPLES)
def test_memref_roundtrip(src: str) -> None:
    assert is_parse_valid(src), f"memref sample rejected: {src!r}"


def test_invalid_rejected() -> None:
    # Missing colon + return type should be rejected.
    bad = "module { func.func @x() { return } }"
    # The grammar permits empty return, so this specific case may parse.
    # Use a structurally invalid sample instead:
    bad = "module { @not_a_function }"
    assert not is_parse_valid(bad)
