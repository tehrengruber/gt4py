from __future__ import annotations

import enum
import textwrap
from typing import Final, TypeAlias, Union

import functional.ffront.field_operator_ast as foast
from eve import codegen
from eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from functional.ffront import common_types


PropertyIdentifier: TypeAlias = Union[type[foast.LocatedNode], tuple[type[foast.LocatedNode], str]]

# see https://docs.python.org/3/reference/expressions.html#operator-precedence
PRECEDENCE: Final[dict[PropertyIdentifier, int]] = {
    # Binding or parenthesized expression, list display, dictionary display, set display
    foast.TupleExpr: 18,
    # Subscription, slicing, call, attribute reference
    foast.Subscript: 17,
    foast.Call: 17,
    foast.Attribute: 17,
    # Await expression
    #  not used
    # Exponentiation
    (foast.BinOp, "power"): 16,
    # Positive, negative, bitwise NOT
    (foast.UnaryOp, "plus"): 14,
    (foast.UnaryOp, "minus"): 14,
    (foast.UnaryOp, "not_"): 14,
    # Multiplication, matrix multiplication, division, floor division, remainder
    (foast.BinOp, "multiplies"): 13,
    (foast.BinOp, "divides"): 13,
    (foast.BinOp, "floordiv"): 13,
    # (foast.BinOp, "remainder"): 13,
    # Addition and subtraction
    (foast.BinOp, "plus"): 12,
    (foast.BinOp, "minus"): 12,
    # Shifts
    #  not used
    # Bitwise AND
    #  not used
    # Bitwise XOR
    #  not used
    # Bitwise OR
    #  not used
    # Comparisons, including membership tests and identity tests
    (foast.Compare, "less"): 7,
    (foast.Compare, "less_equal"): 7,
    (foast.Compare, "greater"): 7,
    (foast.Compare, "greater_equal"): 7,
    (foast.Compare, "not_eq"): 7,
    (foast.Compare, "eq"): 7,
    # Boolean NOT
    (foast.UnaryOp, "not_"): 6,
    # Boolean AND
    (foast.BinOp, "and_"): 5,
    # Boolean OR
    (foast.BinOp, "or_"): 4,
    # Conditional expression
    foast.TernaryExpr: 3,
    # Lambda expression
    #  not used
    # Assignment expression
    #  not used
}

RIGHT_GROUPING = {(foast.BinOp, "power"), foast.TernaryExpr}


class Group(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


def _property_identifier(node: foast.LocatedNode) -> PropertyIdentifier:
    if isinstance(node, foast.BinOp):
        return (foast.BinOp, node.op)
    elif isinstance(node, foast.UnaryOp):
        return (foast.UnaryOp, node.op)
    elif isinstance(node, foast.Compare):
        return (foast.Compare, node.op)

    return type(node)


class PrettyPrinter(codegen.TemplatedGenerator):
    Symbol = as_fmt("{id}")

    Name = as_fmt("{id}")

    Constant = as_fmt("{value}")

    Subscript = as_fmt("{value}[{index}]")

    Attribute = as_fmt("{value}.{attr}")

    TupleExpr = as_fmt("({', '.join(elts)}{',' if len(elts)==1 else ''})")

    UnaryOp = as_fmt("{op}{operand}")

    def visit_BinOp(self, node: foast.BinOp) -> str:
        left = self._parenthesize(node.left, node, Group.LEFT)
        right = self._parenthesize(node.right, node, Group.RIGHT)
        return f"{left} {node.op} {right}"

    def visit_Compare(self, node: foast.Compare) -> str:
        left = self._parenthesize(node.left, node, Group.LEFT)
        right = self._parenthesize(node.right, node, Group.RIGHT)
        return f"{left} {node.op} {right}"

    def visit_TernaryExpr(self, node: foast.TernaryExpr) -> str:
        cond = self.visit(node.condition)
        true_expr = self._parenthesize(node.true_expr, node, Group.LEFT)
        false_expr = self._parenthesize(node.false_expr, node, Group.RIGHT)
        return f"{true_expr} if {cond} else {false_expr}"

    Call = as_fmt("{func}({', '.join(args)})")  # TODO: kwargs

    Assign = as_fmt("{target} = {value}")

    Return = as_fmt("return {value}")

    FunctionDefinition = as_mako(
        textwrap.dedent(
            """
            def ${id}(${', '.join(params_annotated)})${return_type}:
              ${'\\n  '.join(body)}
            """
        ).strip()
    )

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition):
        params = self.visit(node.params)
        types = [
            str(param.type) if not isinstance(param.type, common_types.DeferredSymbolType) else None
            for param in node.params
        ]
        params_annotated = [
            f"{param}: {type_}" if type_ else param for param, type_ in zip(params, types)
        ]
        return_type = (
            f" -> {node.type.returns}"
            if not isinstance(node.type, common_types.DeferredSymbolType)
            else ""
        )
        return self.generic_visit(node, params_annotated=params_annotated, return_type=return_type)

    FieldOperator = as_fmt("@field_operator\n{definition}")

    ScanOperator = as_fmt(
        "@scan_operator(axis={axis}, forward={forward}, init={init})\n{definition}"
    )

    def _precedence(self, node: foast.LocatedNode) -> int:
        prop_id = _property_identifier(node)
        if prop_id in PRECEDENCE:
            return PRECEDENCE[prop_id]
        return max(PRECEDENCE.values()) + 1

    def _grouping(self, node: foast.LocatedNode) -> Group:
        prop_id = _property_identifier(node)
        return Group.RIGHT if prop_id in RIGHT_GROUPING else Group.LEFT

    def _parenthesize(
        self,
        inner_node: foast.LocatedNode,
        outer_node: foast.LocatedNode,
        group: Group | None = None,
    ) -> str:
        inner_precedence = self._precedence(inner_node)
        outer_precedence = self._precedence(outer_node)
        parenthesize = False
        if inner_precedence < outer_precedence:
            parenthesize = True
        elif group is not None and inner_precedence == outer_precedence:
            parenthesize = group != self._grouping(inner_node)
        inner_node_str = self.visit(inner_node)
        return f"({inner_node_str})" if parenthesize else inner_node_str


def pretty_format(node: foast.LocatedNode) -> str:
    """
    Pretty print (to string) an `foast.LocatedNode`.

    >>> from functional.common import Field
    >>> from functional.ffront.decorator import field_operator
    >>> @field_operator
    ... def field_op(a: Field[..., int]):
    ...     return a+1
    >>> print(pretty_format(field_op.foast_node))
    @field_operator
    def field_op(a):
      return a + 1
    """
    return PrettyPrinter().visit(node)