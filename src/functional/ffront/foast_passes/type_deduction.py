# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Optional, cast

import functional.ffront.field_operator_ast as foast
from eve import NodeTranslator, NodeVisitor, traits
from functional.common import DimensionKind, GTSyntaxError, GTTypeError
from functional.ffront import common_types as ct, fbuiltins, type_info
from functional.ffront.foast_passes.utils import compute_assign_indices
from functional.ffront.symbol_makers import make_symbol_type_from_value


def boolified_type(symbol_type: ct.SymbolType) -> ct.ScalarType | ct.FieldType:
    """
    Create a new symbol type from a symbol type, replacing the data type with ``bool``.

    Examples:
    ---------
    >>> from functional.common import Dimension
    >>> scalar_t = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    >>> print(boolified_type(scalar_t))
    bool

    >>> field_t = ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind))
    >>> print(boolified_type(field_t))
    Field[[I], bool]
    """
    shape = None
    if type_info.is_concrete(symbol_type):
        shape = type_info.extract_dtype(symbol_type).shape
    scalar_bool = ct.ScalarType(kind=ct.ScalarKind.BOOL, shape=shape)
    type_class = type_info.type_class(symbol_type)
    if type_class is ct.ScalarType:
        return scalar_bool
    elif type_class is ct.FieldType:
        return ct.FieldType(dtype=scalar_bool, dims=type_info.extract_dims(symbol_type))
    raise GTTypeError(f"Can not boolify type {symbol_type}!")


def construct_tuple_type(
    true_branch_types: list,
    false_branch_types: list,
    mask_type: ct.FieldType,
) -> list:
    """
    Recursively construct  the return types for the tuple return branch.

    Examples:
    ---------
    >>> from functional.common import Dimension
    >>> mask_type = ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind.BOOL))
    >>> true_branch_types = [ct.ScalarType(kind=ct.ScalarKind), ct.ScalarType(kind=ct.ScalarKind)]
    >>> false_branch_types = [ct.FieldType(dims=[Dimension(value="I")], dtype=ct.ScalarType(kind=ct.ScalarKind)), ct.ScalarType(kind=ct.ScalarKind)]
    >>> print(construct_tuple_type(true_branch_types, false_branch_types, mask_type))
    [FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<enum 'ScalarKind'>, shape=None)), FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<enum 'ScalarKind'>, shape=None))]
    """
    element_types_new = true_branch_types
    for i, element in enumerate(true_branch_types):
        if isinstance(element, ct.TupleType):
            element_types_new[i] = ct.TupleType(
                types=construct_tuple_type(element.types, false_branch_types[i].types, mask_type)
            )
        else:
            element_types_new[i] = promote_to_mask_type(
                mask_type, type_info.promote(element_types_new[i], false_branch_types[i])
            )
    return element_types_new


def promote_to_mask_type(
    mask_type: ct.FieldType, input_type: ct.FieldType | ct.ScalarType
) -> ct.FieldType:
    """
    Promote mask type with the input type.

    The input type being the result of promoting the left and right types in a conditional clause.

    If the input type is a scalar, the return type takes the dimensions of the mask_type, while retaining the dtype of
    the input type. The behavior is similar when the input type is a field type with fewer dimensions than the mask_type.
    In all other cases, the return type takes the dimensions and dtype of the input type.

    >>> from functional.common import Dimension
    >>> I, J = (Dimension(value=dim) for dim in ["I", "J"])
    >>> bool_type = ct.ScalarType(kind=ct.ScalarKind.BOOL)
    >>> dtype = ct.ScalarType(kind=ct.ScalarKind.FLOAT64)
    >>> promote_to_mask_type(ct.FieldType(dims=[I, J], dtype=bool_type), ct.ScalarType(kind=dtype))
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None), shape=None))
    >>> promote_to_mask_type(ct.FieldType(dims=[I, J], dtype=bool_type), ct.FieldType(dims=[I], dtype=dtype))
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None))
    >>> promote_to_mask_type(ct.FieldType(dims=[I], dtype=bool_type), ct.FieldType(dims=[I,J], dtype=dtype))
    FieldType(dims=[Dimension(value='I', kind=<DimensionKind.HORIZONTAL: 'horizontal'>), Dimension(value='J', kind=<DimensionKind.HORIZONTAL: 'horizontal'>)], dtype=ScalarType(kind=<ScalarKind.FLOAT64: 1064>, shape=None))
    """
    if isinstance(input_type, ct.ScalarType) or not all(
        item in input_type.dims for item in mask_type.dims
    ):
        return_dtype = input_type.dtype if isinstance(input_type, ct.FieldType) else input_type
        return type_info.promote(input_type, ct.FieldType(dims=mask_type.dims, dtype=return_dtype))  # type: ignore
    else:
        return input_type


class FieldOperatorTypeDeductionCompletnessValidator(NodeVisitor):
    """Validate an FOAST expression is fully typed."""

    @classmethod
    def apply(cls, node: foast.LocatedNode) -> None:
        incomplete_nodes: list[foast.LocatedNode] = []
        cls().visit(node, incomplete_nodes=incomplete_nodes)

        if incomplete_nodes:
            raise AssertionError("FOAST expression is not fully typed.")

    def visit_LocatedNode(
        self, node: foast.LocatedNode, *, incomplete_nodes: list[foast.LocatedNode]
    ):
        self.generic_visit(node, incomplete_nodes=incomplete_nodes)

        if hasattr(node, "type") and not type_info.is_concrete(node.type):
            incomplete_nodes.append(node)


class FieldOperatorTypeDeduction(traits.VisitorWithSymbolTableTrait, NodeTranslator):
    """
    Deduce and check types of FOAST expressions and symbols.

    Examples:
    ---------
    >>> import ast
    >>> import typing
    >>> from functional.common import Field
    >>> from functional.ffront.source_utils import SourceDefinition, get_closure_vars_from_function
    >>> from functional.ffront.func_to_foast import FieldOperatorParser
    >>> def example(a: "Field[..., float]", b: "Field[..., float]"):
    ...     return a + b

    >>> source_definition = SourceDefinition.from_function(example)
    >>> closure_vars = get_closure_vars_from_function(example)
    >>> annotations = typing.get_type_hints(example)
    >>> untyped_fieldop = FieldOperatorParser(
    ...     source_definition=source_definition, closure_vars=closure_vars, annotations=annotations
    ... ).visit(ast.parse(source_definition.source).body[0])
    >>> untyped_fieldop.body[0].value.type
    DeferredSymbolType(constraint=None)

    >>> typed_fieldop = FieldOperatorTypeDeduction.apply(untyped_fieldop)
    >>> assert typed_fieldop.body[0].value.type == ct.FieldType(dtype=ct.ScalarType(
    ...     kind=ct.ScalarKind.FLOAT64), dims=Ellipsis)
    """

    @classmethod
    def apply(cls, node: foast.FieldOperator) -> foast.FieldOperator:
        typed_foast_node = cls().visit(node)

        FieldOperatorTypeDeductionCompletnessValidator.apply(typed_foast_node)

        return typed_foast_node

    def visit_FunctionDefinition(self, node: foast.FunctionDefinition, **kwargs):
        new_params = self.visit(node.params, **kwargs)
        new_body = self.visit(node.body, **kwargs)
        new_closure_vars = self.visit(node.closure_vars, **kwargs)
        return_type = self._deduce_return_type(new_body, **kwargs)
        new_type = ct.FunctionType(
            args=[new_param.type for new_param in new_params], kwargs={}, returns=return_type
        )
        return foast.FunctionDefinition(
            id=node.id,
            params=new_params,
            body=new_body,
            closure_vars=new_closure_vars,
            type=new_type,
            location=node.location,
        )

    def _deduce_return_type(
        self, node: foast.BlockStmt, *, requires_unconditional_return=True, **kwargs
    ):
        conditional_return_type: ct.DataType | None = None

        for stmt in node.stmts:
            is_unconditional_return = False

            if isinstance(stmt, foast.Return):
                is_unconditional_return = True
                return_type = stmt.value.type
            elif isinstance(stmt, foast.IfStmt):
                return_types = (
                    self._deduce_return_type(stmt.true_branch, requires_unconditional_return=False),
                    self._deduce_return_type(
                        stmt.false_branch, requires_unconditional_return=False
                    ),
                )
                # if both branches return
                if return_types[0] and return_types[1]:
                    if return_types[0] == return_types[1]:
                        is_unconditional_return = True
                    else:
                        # types must match
                        raise AssertionError()
                return_type = return_types[0] or return_types[1]
            elif isinstance(stmt, foast.BlockStmt):
                # just forward to nested BlockStmt
                return_type = self._deduce_return_type(
                    stmt, requires_unconditional_return=requires_unconditional_return
                )
            elif isinstance(stmt, (foast.Assign, foast.TupleTargetAssign)):
                return_type = None
            else:
                breakpoint()
                raise AssertionError()

            if conditional_return_type and return_type and return_type != conditional_return_type:
                raise AssertionError()

            if is_unconditional_return:  # found a statement that always returns
                assert return_type
                return return_type
            elif return_type:
                conditional_return_type = return_type

        if requires_unconditional_return:
            raise AssertionError()

        return None

    # TODO(tehrengruber): make sure all scalar arguments are lifted to 0-dim field args
    def visit_FieldOperator(self, node: foast.FieldOperator, **kwargs) -> foast.FieldOperator:
        new_definition = self.visit(node.definition, **kwargs)
        return foast.FieldOperator(
            id=node.id,
            definition=new_definition,
            location=node.location,
            type=ct.FieldOperatorType(definition=new_definition.type),
        )

    def visit_ScanOperator(self, node: foast.ScanOperator, **kwargs) -> foast.ScanOperator:
        new_axis = self.visit(node.axis, **kwargs)
        if not isinstance(new_axis.type, ct.DimensionType):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `axis` to scan operator `{node.id}` must be a dimension.",
            )
        if not new_axis.type.dim.kind == DimensionKind.VERTICAL:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `axis` to scan operator `{node.id}` must be a vertical dimension.",
            )
        new_forward = self.visit(node.forward, **kwargs)
        if not new_forward.type.kind == ct.ScalarKind.BOOL:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Argument `forward` to scan operator `{node.id}` must" f"be a boolean."
            )
        new_init = self.visit(node.init, **kwargs)
        if not all(
            type_info.is_arithmetic(type_) or type_info.is_logical(type_)
            for type_ in type_info.primitive_constituents(new_init.type)
        ):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Argument `init` to scan operator `{node.id}` must "
                f"be an arithmetic type or a logical type or a composite of arithmetic and logical types.",
            )
        new_definition = self.visit(node.definition, **kwargs)
        new_type = ct.ScanOperatorType(
            axis=new_axis.type.dim,
            definition=new_definition.type,
        )
        return foast.ScanOperator(
            id=node.id,
            axis=new_axis,
            forward=new_forward,
            init=new_init,
            definition=new_definition,
            type=new_type,
            location=node.location,
        )

    def visit_Name(self, node: foast.Name, **kwargs) -> foast.Name:
        symtable = kwargs["symtable"]
        if node.id not in symtable or symtable[node.id].type is None:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Undeclared symbol `{node.id}`."
            )

        symbol = symtable[node.id]
        return foast.Name(id=node.id, type=symbol.type, location=node.location)

    def visit_Assign(self, node: foast.Assign, **kwargs) -> foast.Assign:
        new_value = node.value
        if not type_info.is_concrete(node.value.type):
            new_value = self.visit(node.value, **kwargs)
        new_target = self.visit(node.target, refine_type=new_value.type, **kwargs)
        return foast.Assign(target=new_target, value=new_value, location=node.location)

    def visit_TupleTargetAssign(
        self, node: foast.TupleTargetAssign, **kwargs
    ) -> foast.TupleTargetAssign:

        TargetType = list[foast.Starred | foast.Symbol]
        values = self.visit(node.value, **kwargs)

        if isinstance(values.type, ct.TupleType):
            num_elts: int = len(values.type.types)
            targets: TargetType = node.targets
            indices: list[tuple[int, int] | int] = compute_assign_indices(targets, num_elts)

            if not any(isinstance(i, tuple) for i in indices) and len(indices) != num_elts:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg=f"Too many values to unpack (expected {len(indices)})."
                )

            new_targets: TargetType = []
            new_type: ct.TupleType | ct.DataType
            for i, index in enumerate(indices):
                old_target = targets[i]

                if isinstance(index, tuple):
                    lower, upper = index
                    new_type = ct.TupleType(types=[t for t in values.type.types[lower:upper]])
                    new_target = foast.Starred(
                        id=foast.DataSymbol(
                            id=self.visit(old_target.id).id,
                            location=old_target.location,
                            type=new_type,
                        ),
                        type=new_type,
                        location=old_target.location,
                    )
                else:
                    new_type = values.type.types[index]
                    new_target = self.visit(
                        old_target, refine_type=new_type, location=old_target.location, **kwargs
                    )

                new_target = self.visit(
                    new_target, refine_type=new_type, location=old_target.location, **kwargs
                )
                new_targets.append(new_target)
        else:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Assignment value must be of type tuple! Got: {values.type}"
            )

        return foast.TupleTargetAssign(targets=new_targets, value=values, location=node.location)


    def visit_IfStmt(self, node: foast.IfStmt, **kwargs):
        symtable = kwargs["symtable"]

        new_true_branch = self.visit(node.true_branch, **kwargs)
        new_false_branch = self.visit(node.false_branch, **kwargs)
        new_node = foast.IfStmt(
            condition=self.visit(node.condition, **kwargs),
            true_branch=new_true_branch,
            false_branch=new_false_branch,
            location=node.location,
        )

        if not isinstance(new_node.condition.type, ct.ScalarType):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg="Condition for `if` must be scalar. "
                f"But got `{new_node.condition.type}` instead.",
            )

        if new_node.condition.type.kind != ct.ScalarKind.BOOL:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg="Condition for `if` must be of boolean type. "
                f"But got `{new_node.condition.type}` instead.",
            )

        for sym in node.annex._common_symbols.keys():
            if (true_type := new_true_branch.annex.symtable[sym].type) != (
                false_type := new_false_branch.annex.symtable[sym].type
            ):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=f"Inconsistent types between two branches for variable `{sym}`. "
                    f"Got types `{true_type}` and `{false_type}.",
                )
            # TODO: properly patch symtable (new node?)
            symtable[sym].type = new_node.annex._common_symbols[
                sym
            ].type = new_true_branch.annex.symtable[sym].type

        return new_node

    def visit_Symbol(
        self,
        node: foast.Symbol,
        refine_type: Optional[ct.FieldType] = None,
        **kwargs,
    ) -> foast.Symbol:
        symtable = kwargs["symtable"]
        if refine_type:
            if not type_info.is_concretizable(node.type, to_type=refine_type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=(
                        "type inconsistency: expression was deduced to be "
                        f"of type {refine_type}, instead of the expected type {node.type}"
                    ),
                )
            new_node: foast.Symbol = foast.Symbol(
                id=node.id, type=refine_type, location=node.location
            )
            symtable[new_node.id] = new_node
            return new_node
        return node

    def visit_Subscript(self, node: foast.Subscript, **kwargs) -> foast.Subscript:
        new_value = self.visit(node.value, **kwargs)
        new_type: Optional[ct.SymbolType] = None
        match new_value.type:
            case ct.TupleType(types=types):
                new_type = types[node.index]
            case ct.OffsetType(source=source, target=(target1, target2)):
                if not target2.kind == DimensionKind.LOCAL:
                    raise FieldOperatorTypeDeductionError.from_foast_node(
                        new_value, msg="Second dimension in offset must be a local dimension."
                    )
                new_type = ct.OffsetType(source=source, target=(target1,))
            case ct.OffsetType(source=source, target=(target,)):
                # for cartesian axes (e.g. I, J) the index of the subscript only
                #  signifies the displacement in the respective dimension,
                #  but does not change the target type.
                if source != target:
                    raise FieldOperatorTypeDeductionError.from_foast_node(
                        new_value,
                        msg="Source and target must be equal for offsets with a single target.",
                    )
                new_type = new_value.type
            case _:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    new_value, msg="Could not deduce type of subscript expression!"
                )

        return foast.Subscript(
            value=new_value, index=node.index, type=new_type, location=node.location
        )

    def visit_BinOp(self, node: foast.BinOp, **kwargs) -> foast.BinOp:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_binop_type(node, left=new_left, right=new_right)
        return foast.BinOp(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def visit_TernaryExpr(self, node: foast.TernaryExpr, **kwargs) -> foast.TernaryExpr:
        new_condition = self.visit(node.condition, **kwargs)
        new_true_expr = self.visit(node.true_expr, **kwargs)
        new_false_expr = self.visit(node.false_expr, **kwargs)
        new_type = self._deduce_ternaryexpr_type(
            node, condition=new_condition, true_expr=new_true_expr, false_expr=new_false_expr
        )
        return foast.TernaryExpr(
            condition=new_condition,
            true_expr=new_true_expr,
            false_expr=new_false_expr,
            location=node.location,
            type=new_type,
        )

    def _deduce_ternaryexpr_type(
        self,
        node: foast.TernaryExpr,
        *,
        condition: foast.Expr,
        true_expr: foast.Expr,
        false_expr: foast.Expr,
    ) -> Optional[ct.SymbolType]:
        if condition.type != ct.ScalarType(kind=ct.ScalarKind.BOOL):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                condition,
                msg=f"Condition is of type `{condition.type}` " f"but should be of type bool.",
            )

        if true_expr.type != false_expr.type:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Left and right types are not the same: `{true_expr.type}` and `{false_expr.type}`",
            )
        return true_expr.type

    def visit_Compare(self, node: foast.Compare, **kwargs) -> foast.Compare:
        new_left = self.visit(node.left, **kwargs)
        new_right = self.visit(node.right, **kwargs)
        new_type = self._deduce_compare_type(node, left=new_left, right=new_right)
        return foast.Compare(
            op=node.op, left=new_left, right=new_right, location=node.location, type=new_type
        )

    def _deduce_compare_type(
        self, node: foast.Compare, *, left: foast.Expr, right: foast.Expr, **kwargs
    ) -> Optional[ct.SymbolType]:
        # check both types compatible
        for arg in (left, right):
            if not type_info.is_arithmetic(arg.type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    arg, msg=f"Type {arg.type} can not be used in operator '{node.op}'!"
                )

        self._check_operand_dtypes_match(node, left=left, right=right)

        try:
            # transform operands to have bool dtype and use regular promotion
            #  mechanism to handle dimension promotion
            return type_info.promote(boolified_type(left.type), boolified_type(right.type))
        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Could not promote `{left.type}` and `{right.type}` to common type"
                f" in call to `{node.op}`.",
            ) from ex

    def _deduce_binop_type(
        self,
        node: foast.BinOp,
        *,
        left: foast.Expr,
        right: foast.Expr,
        **kwargs,
    ) -> Optional[ct.SymbolType]:
        logical_ops = {
            ct.BinaryOperator.BIT_AND,
            ct.BinaryOperator.BIT_OR,
            ct.BinaryOperator.BIT_XOR,
        }
        is_compatible = type_info.is_logical if node.op in logical_ops else type_info.is_arithmetic

        # check both types compatible
        for arg in (left, right):
            if not is_compatible(arg.type):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    arg, msg=f"Type {arg.type} can not be used in operator `{node.op}`!"
                )

        left_type = cast(ct.FieldType | ct.ScalarType, left.type)
        right_type = cast(ct.FieldType | ct.ScalarType, right.type)

        if node.op == ct.BinaryOperator.POW:
            return left_type

        try:
            return type_info.promote(left_type, right_type)
        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Could not promote `{left_type}` and `{right_type}` to common type"
                f" in call to `{node.op}`.",
            ) from ex

    def _check_operand_dtypes_match(
        self, node: foast.BinOp | foast.Compare, left: foast.Expr, right: foast.Expr
    ) -> None:
        # check dtypes match
        if not type_info.extract_dtype(left.type) == type_info.extract_dtype(right.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible datatypes in operator `{node.op}`: {left.type} and {right.type}!",
            )

    def visit_UnaryOp(self, node: foast.UnaryOp, **kwargs) -> foast.UnaryOp:
        new_operand = self.visit(node.operand, **kwargs)
        is_compatible = (
            type_info.is_logical
            if node.op in [ct.UnaryOperator.NOT, ct.UnaryOperator.INVERT]
            else type_info.is_arithmetic
        )
        if not is_compatible(new_operand.type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible type for unary operator `{node.op}`: `{new_operand.type}`!",
            )
        return foast.UnaryOp(
            op=node.op, operand=new_operand, location=node.location, type=new_operand.type
        )

    def visit_TupleExpr(self, node: foast.TupleExpr, **kwargs) -> foast.TupleExpr:
        new_elts = self.visit(node.elts, **kwargs)
        new_type = ct.TupleType(types=[element.type for element in new_elts])
        return foast.TupleExpr(elts=new_elts, type=new_type, location=node.location)

    def visit_Call(self, node: foast.Call, **kwargs) -> foast.Call:
        new_func = self.visit(node.func, **kwargs)
        new_args = self.visit(node.args, **kwargs)
        new_kwargs = self.visit(node.kwargs, **kwargs)

        func_type = new_func.type
        arg_types = [arg.type for arg in new_args]
        kwarg_types = {name: arg.type for name, arg in new_kwargs.items()}

        # ensure signature is valid
        try:
            type_info.accepts_args(
                func_type,
                with_args=arg_types,
                with_kwargs=kwarg_types,
                raise_exception=True,
            )
        except GTTypeError as err:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Invalid argument types in call to `{node.func.id}`!"
            ) from err

        return_type = type_info.return_type(func_type, with_args=arg_types, with_kwargs=kwarg_types)

        new_node = foast.Call(
            func=new_func,
            args=new_args,
            kwargs=new_kwargs,
            location=node.location,
            type=return_type,
        )

        if (
            isinstance(new_func.type, ct.FunctionType)
            and new_func.id in fbuiltins.MATH_BUILTIN_NAMES
        ):
            return self._visit_math_built_in(new_node, **kwargs)
        elif (
            isinstance(new_func.type, ct.FunctionType)
            and not type_info.is_concrete(return_type)
            and new_func.id in fbuiltins.FUN_BUILTIN_NAMES
        ):
            visitor = getattr(self, f"_visit_{new_func.id}")
            return visitor(new_node, **kwargs)

        return new_node

    def _ensure_signature_valid(self, node: foast.Call, **kwargs) -> None:
        try:
            type_info.accepts_args(
                cast(ct.FunctionType, node.func.type),
                with_args=[arg.type for arg in node.args],
                with_kwargs={keyword: arg.type for keyword, arg in node.kwargs.items()},
                raise_exception=True,
            )
        except GTTypeError as err:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg=f"Invalid argument types in call to `{node.func.id}`!"
            ) from err

    def _visit_math_built_in(self, node: foast.Call, **kwargs) -> foast.Call:
        func_name = node.func.id

        # validate arguments
        error_msg_preamble = f"Incompatible argument in call to `{func_name}`."
        error_msg_for_validator = {
            type_info.is_arithmetic: "an arithmetic",
            type_info.is_floating_point: "a floating point",
        }
        if func_name in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        elif func_name in fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            arg_validator = type_info.is_floating_point
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            arg_validator = type_info.is_arithmetic
        else:
            raise AssertionError(f"Unknown math builtin `{func_name}`.")

        error_msgs = []
        for i, arg in enumerate(node.args):
            if not arg_validator(arg.type):
                error_msgs.append(
                    f"Expected {i}-th argument to be {error_msg_for_validator[arg_validator]} type, but got `{arg.type}`."
                )
        if error_msgs:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg="\n".join([error_msg_preamble] + [f"  - {error}" for error in error_msgs]),
            )

        if func_name == "power" and all(type_info.is_integral(arg.type) for arg in node.args):
            print(f"Warning: return type of {func_name} might be inconsistent (not implemented).")

        # deduce return type
        return_type: Optional[ct.FieldType | ct.ScalarType] = None
        if (
            func_name
            in fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES + fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
        ):
            return_type = cast(ct.FieldType | ct.ScalarType, node.args[0].type)
        elif func_name in fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES:
            return_type = boolified_type(cast(ct.FieldType | ct.ScalarType, node.args[0].type))
        elif func_name in fbuiltins.BINARY_MATH_NUMBER_BUILTIN_NAMES:
            try:
                return_type = type_info.promote(
                    *((cast(ct.FieldType | ct.ScalarType, arg.type)) for arg in node.args)
                )
            except GTTypeError as ex:
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node, msg=error_msg_preamble
                ) from ex
        else:
            raise AssertionError(f"Unknown math builtin `{func_name}`.")

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_reduction(self, node: foast.Call, **kwargs) -> foast.Call:
        field_type = cast(ct.FieldType, node.args[0].type)
        reduction_dim = cast(ct.DimensionType, node.kwargs["axis"].type).dim
        if reduction_dim not in field_type.dims:
            field_dims_str = ", ".join(str(dim) for dim in field_type.dims)
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible field argument in call to `{node.func.id}`. "
                f"Expected a field with dimension {reduction_dim}, but got "
                f"{field_dims_str}.",
            )
        return_type = ct.FieldType(
            dims=[dim for dim in field_type.dims if dim != reduction_dim],
            dtype=field_type.dtype,
        )

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def _visit_neighbor_sum(self, node: foast.Call, **kwargs) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_max_over(self, node: foast.Call, **kwargs) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_min_over(self, node: foast.Call, **kwargs) -> foast.Call:
        return self._visit_reduction(node, **kwargs)

    def _visit_where(self, node: foast.Call, **kwargs) -> foast.Call:
        mask_type = cast(ct.FieldType, node.args[0].type)
        true_branch_type = node.args[1].type
        false_branch_type = node.args[2].type
        return_type: ct.TupleType | ct.FieldType
        if not type_info.is_logical(mask_type):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible argument in call to `{node.func.id}`. Expected "
                f"a field with dtype bool, but got `{mask_type}`.",
            )

        try:
            if isinstance(true_branch_type, ct.TupleType) and isinstance(
                false_branch_type, ct.TupleType
            ):
                return_type = ct.TupleType(
                    types=construct_tuple_type(
                        true_branch_type.types, false_branch_type.types, mask_type
                    )
                )
            elif isinstance(true_branch_type, ct.TupleType) or isinstance(
                false_branch_type, ct.TupleType
            ):
                raise FieldOperatorTypeDeductionError.from_foast_node(
                    node,
                    msg=f"Return arguments need to be of same type in {node.func.id}, but got: "
                    f"{node.args[1].type} and {node.args[2].type}",
                )
            else:
                true_branch_fieldtype = cast(ct.FieldType, true_branch_type)
                false_branch_fieldtype = cast(ct.FieldType, false_branch_type)
                promoted_type = type_info.promote(true_branch_fieldtype, false_branch_fieldtype)
                return_type = promote_to_mask_type(mask_type, promoted_type)

        except GTTypeError as ex:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible argument in call to `{node.func.id}`.",
            ) from ex

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            type=return_type,
            location=node.location,
        )

    def _visit_broadcast(self, node: foast.Call, **kwargs) -> foast.Call:
        arg_type = cast(ct.FieldType | ct.ScalarType, node.args[0].type)
        broadcast_dims_expr = cast(foast.TupleExpr, node.args[1]).elts

        if any([not (isinstance(elt.type, ct.DimensionType)) for elt in broadcast_dims_expr]):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimension type in {node.func.id}. Expected "
                f"all broadcast dimensions to be of type Dimension.",
            )

        broadcast_dims = [cast(ct.DimensionType, elt.type).dim for elt in broadcast_dims_expr]

        if not set((arg_dims := type_info.extract_dims(arg_type))).issubset(set(broadcast_dims)):
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node,
                msg=f"Incompatible broadcast dimensions in {node.func.id}. Expected "
                f"broadcast dimension is missing {set(arg_dims).difference(set(broadcast_dims))}",
            )

        return_type = ct.FieldType(
            dims=broadcast_dims,
            dtype=type_info.extract_dtype(arg_type),
        )

        return foast.Call(
            func=node.func,
            args=node.args,
            kwargs=node.kwargs,
            location=node.location,
            type=return_type,
        )

    def visit_Constant(self, node: foast.Constant, **kwargs) -> foast.Constant:
        try:
            type_ = make_symbol_type_from_value(node.value)
        except GTTypeError as e:
            raise FieldOperatorTypeDeductionError.from_foast_node(
                node, msg="Could not deduce type of constant."
            ) from e
        return foast.Constant(value=node.value, location=node.location, type=type_)


class FieldOperatorTypeDeductionError(GTSyntaxError, SyntaxWarning):
    """Exception for problematic type deductions that originate in user code."""

    def __init__(
        self,
        msg="",
        *,
        lineno=0,
        offset=0,
        filename=None,
        end_lineno=None,
        end_offset=None,
        text=None,
    ):
        msg = "Could not deduce type: " + msg
        super().__init__(msg, (filename, lineno, offset, text, end_lineno, end_offset))

    @classmethod
    def from_foast_node(
        cls,
        node: foast.LocatedNode,
        *,
        msg: str = "",
    ):
        return cls(
            msg,
            lineno=node.location.line,
            offset=node.location.column,
            filename=node.location.source,
            end_lineno=node.location.end_line,
            end_offset=node.location.end_column,
        )
