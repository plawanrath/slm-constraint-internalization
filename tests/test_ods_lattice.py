"""Tests for ODS .td parsing + lattice construction."""
from __future__ import annotations

from sci.constraints.ods_parse import parse_td
from sci.constraints.ods_lattice import build_entry, expand_class, ANY


SAMPLE_TD = '''
def Arith_AddIOp : Arith_IntBinaryOp<"addi", [SameOperandsAndResultType]> {
  let summary = "integer addition operation";
  let description = [{
    The `addi` operation takes two operands and returns one result, each of
    these is required to be of the same type.
  }];
  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger:$result);
}

def Arith_ConstantOp : Arith_Op<"constant", [ConstantLike, Pure]> {
  let summary = "integer or floating point constant";
  let description = [{ ... }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
}
'''


def test_parse_two_ops() -> None:
    ops = parse_td(SAMPLE_TD)
    assert len(ops) == 2
    names = {op.full_name for op in ops}
    assert "arith.addi" in names
    assert "arith.constant" in names


def test_addi_lattice() -> None:
    (addi,) = [op for op in parse_td(SAMPLE_TD) if op.mnemonic == "addi"]
    entry = build_entry(addi)
    assert entry.operand_arity == (2, 2)
    assert entry.result_arity == (1, 1)
    # AnyInteger should expand to primitive signless ints.
    assert "i32" in entry.operand_types[0]
    assert "i64" in entry.operand_types[1]
    # SameOperandsAndResultType should produce a same-type class.
    assert len(entry.same_type_classes) == 1


def test_constant_lattice_has_any_result() -> None:
    (const,) = [op for op in parse_td(SAMPLE_TD) if op.mnemonic == "constant"]
    entry = build_entry(const)
    assert entry.result_types == [[ANY]]


def test_expand_unknown_class_returns_any() -> None:
    assert expand_class("SomeCompletelyUnknownClass") == [ANY]


def test_expand_known_classes() -> None:
    assert "f32" in expand_class("AnyFloat")
    assert "memref" in expand_class("AnyMemRef")
