# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
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
import collections.abc
import copy
import dataclasses
import functools
import inspect

from gt4py import eve
from gt4py.eve import concepts
from gt4py.eve.extended_typing import Any, Callable, Optional, TypeVar, Union
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils.common_pattern_matcher import is_call_to
from gt4py.next.iterator.transforms import global_tmps
from gt4py.next.iterator.type_system import rules, type_specifications as it_ts
from gt4py.next.type_system import type_info, type_specifications as ts


def _is_representable_as_int(s: int | str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def _is_compatible_type(type_a: ts.TypeSpec, type_b: ts.TypeSpec):
    """
    Predicate to determine if two types are compatible.

    This function gracefully handles:
     - iterators with unknown positions which are considered compatible to any other positions
       of another iterator.
     - iterators which are defined everywhere, i.e. empty defined dimensions
    Beside that this function simply checks for equality of types.

    >>> bool_type = ts.ScalarType(kind=ts.ScalarKind.BOOL)
    >>> IDim = common.Dimension(value="IDim")
    >>> type_on_i_of_i_it = it_ts.IteratorType(
    ...     position_dims=[IDim], defined_dims=[IDim], element_type=bool_type
    ... )
    >>> type_on_undefined_of_i_it = it_ts.IteratorType(
    ...     position_dims="unknown", defined_dims=[IDim], element_type=bool_type
    ... )
    >>> _is_compatible_type(type_on_i_of_i_it, type_on_undefined_of_i_it)
    True

    >>> JDim = common.Dimension(value="JDim")
    >>> type_on_j_of_j_it = it_ts.IteratorType(
    ...     position_dims=[JDim], defined_dims=[JDim], element_type=bool_type
    ... )
    >>> _is_compatible_type(type_on_i_of_i_it, type_on_j_of_j_it)
    False
    """
    is_compatible = True

    if isinstance(type_a, it_ts.IteratorType) and isinstance(type_b, it_ts.IteratorType):
        if not any(el_type.position_dims == "unknown" for el_type in [type_a, type_b]):
            is_compatible &= type_a.position_dims == type_b.position_dims
        if type_a.defined_dims and type_b.defined_dims:
            is_compatible &= type_a.defined_dims == type_b.defined_dims
        is_compatible &= type_a.element_type == type_b.element_type
    elif isinstance(type_a, ts.TupleType):
        for el_type_a, el_type_b in zip(type_a.types, type_b.types, strict=True):
            is_compatible &= _is_compatible_type(el_type_a, el_type_b)
    elif isinstance(type_a, ts.FunctionType):
        for arg_a, arg_b in zip(type_a.pos_only_args, type_b.pos_only_args, strict=True):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.pos_or_kw_args.values(), type_b.pos_or_kw_args.values(), strict=True
        ):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        for arg_a, arg_b in zip(
            type_a.kw_only_args.values(), type_b.kw_only_args.values(), strict=True
        ):
            is_compatible &= _is_compatible_type(arg_a, arg_b)
        is_compatible &= _is_compatible_type(type_a.returns, type_b.returns)
    else:
        is_compatible &= type_a == type_b

    return is_compatible


# TODO(tehrengruber): remove after documentation is written
# Problems:
#  - what happens when we get a lambda function whose params are already typed
#  - write back params type in lambda
#  - documentation
#    describe why lambda can only have one type. Describe idea to solve e.g.
#     `let("f", lambda x: x)(f(1)+f(1.))
#     -> let("f_int", lambda x: x, "f_float", lambda x: x)(f_int(1)+f_float(1.))`
#    describe where this is needed, e.g.:
#      `if_(cond, fun_tail(it_on_vertex), fun_tail(it_on_vertex_k))`
#  - document how scans are handled (also mention to Hannes)
#  - types are stored in the node, but will be incomplete after some passes
# Design decisions
#  Only the parameters of fencils need to be typed.
#  Lambda functions are not polymorphic.
def _set_node_type(node: itir.Node, type_: ts.TypeSpec) -> None:
    if node.type:
        assert _is_compatible_type(node.type, type_), "Node already has a type which differs."
    node.type = type_


def on_inferred(
    callback: Callable, *args: Union[ts.TypeSpec, "ObservableTypeInferenceRule"]
) -> None:
    """
    Execute `callback` as soon as all `args` have a type.
    """
    ready_args = [False] * len(args)
    inferred_args = [None] * len(args)

    def mark_ready(i, type_):
        ready_args[i] = True
        inferred_args[i] = type_
        if all(ready_args):
            callback(*inferred_args)

    for i, arg in enumerate(args):
        if isinstance(arg, ObservableTypeInferenceRule):
            arg.on_type_ready(functools.partial(mark_ready, i))
        else:
            assert isinstance(arg, ts.TypeSpec)
            mark_ready(i, arg)


@dataclasses.dataclass
class ObservableTypeInferenceRule:
    """
    This class wraps a raw type inference rule to handle typing of functions.

    TODO: As functions in ITIR are represented by type inference rules, i.e. regular callables,
    """

    #: type rule that given a set of types or type rules returns the return type or a type rule
    type_rule: rules.TypeInferenceRule
    #: offset provider used by some type rules
    offset_provider: common.OffsetProvider
    #: node that has this type
    node: Optional[itir.Node] = None
    #: list of references to this function
    aliases: list[itir.SymRef] = dataclasses.field(default_factory=list)
    #: list of callbacks executed as soon as the type is ready
    callbacks: list[Callable[[ts.TypeSpec], None]] = dataclasses.field(default_factory=list)
    #: the inferred type when ready and None until then
    inferred_type: Optional[ts.FunctionType] = None
    #: whether to store the type in the node or not
    store_inferred_type_in_node: bool = False

    def infer_type(
        self, return_type: ts.DataType | ts.DeferredType, *args: ts.DataType | ts.DeferredType
    ) -> ts.FunctionType:
        return ts.FunctionType(
            pos_only_args=list(args), pos_or_kw_args={}, kw_only_args={}, returns=return_type
        )

    def _infer_type_listener(self, return_type: ts.TypeSpec, *args: ts.TypeSpec) -> None:
        self.inferred_type = self.infer_type(return_type, *args)  # type: ignore[arg-type]  # ensured by assert above

        # if the type has been fully inferred, notify all `ObservableTypeInferenceRule`s that depend on it.
        for cb in self.callbacks:
            cb(self.inferred_type)

        if self.store_inferred_type_in_node:
            assert self.node
            _set_node_type(self.node, self.inferred_type)
            self.node.type = self.inferred_type
            for alias in self.aliases:
                _set_node_type(alias, self.inferred_type)

    def on_type_ready(self, cb: Callable[[ts.TypeSpec], None]) -> None:
        if self.inferred_type:
            # type has already been inferred, just call the callback
            cb(self.inferred_type)
        else:
            self.callbacks.append(cb)

    def __call__(self, *args) -> ts.TypeSpec | rules.TypeInferenceRule:
        if "offset_provider" in inspect.signature(self.type_rule).parameters:
            return_type = self.type_rule(*args, offset_provider=self.offset_provider)
        else:
            return_type = self.type_rule(*args)

        # return type is a typing rule by itself
        if callable(return_type):
            return_type = ObservableTypeInferenceRule(
                node=None,  # node will be set by caller
                type_rule=return_type,
                offset_provider=self.offset_provider,
                store_inferred_type_in_node=True,
            )

        # delay storing the type until the return type and all arguments are inferred
        on_inferred(self._infer_type_listener, return_type, *args)

        return return_type


def _get_dimensions_from_offset_provider(offset_provider) -> dict[str, common.Dimension]:
    dimensions: dict[str, common.Dimension] = {}
    for offset_name, provider in offset_provider.items():
        dimensions[offset_name] = common.Dimension(
            value=offset_name, kind=common.DimensionKind.LOCAL
        )
        if isinstance(provider, common.Dimension):
            dimensions[provider.value] = provider
        elif isinstance(provider, common.Connectivity):
            dimensions[provider.origin_axis.value] = provider.origin_axis
            dimensions[provider.neighbor_axis.value] = provider.neighbor_axis
    return dimensions


def _get_dimensions_from_types(types) -> dict[str, common.Dimension]:
    def _get_dimensions(obj: Any):
        if isinstance(obj, common.Dimension):
            yield obj
        elif isinstance(obj, ts.TypeSpec):
            for field in dataclasses.fields(obj.__class__):
                yield from _get_dimensions(getattr(obj, field.name))
        elif isinstance(obj, collections.abc.Mapping):
            for el in obj.values():
                yield from _get_dimensions(el)
        elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, str):
            for el in obj:
                yield from _get_dimensions(el)

    return {dim.value: dim for dim in _get_dimensions(types)}


T = TypeVar("T", bound=itir.Node)


def _convert_closure_input_to_iterator(
    domain: it_ts.DomainType, input_: ts.TypeSpec
) -> it_ts.IteratorType:
    input_dims: list[common.Dimension] | None = None

    def extract_dtype_and_dims(el_type: ts.TypeSpec):
        nonlocal input_dims
        assert isinstance(el_type, (ts.FieldType, ts.ScalarType))
        el_type = type_info.promote(el_type, always_field=True)
        if not input_dims:
            input_dims = el_type.dims  # type: ignore[union-attr]  # ensured by always_field
        else:
            # tuple inputs must all have the same defined dimensions as we
            # create an iterator of tuples from them
            assert input_dims == el_type.dims  # type: ignore[union-attr]  # ensured by always_field
        return el_type.dtype  # type: ignore[union-attr]  # ensured by always_field

    element_type = type_info.apply_to_primitive_constituents(extract_dtype_and_dims, input_)

    assert input_dims is not None

    # handle neighbor / sparse input fields
    defined_dims = []
    is_nb_field = False
    for dim in input_dims:
        if dim.kind == common.DimensionKind.LOCAL:
            assert not is_nb_field
            is_nb_field = True
        else:
            defined_dims.append(dim)
    if is_nb_field:
        element_type = it_ts.ListType(element_type=element_type)

    return it_ts.IteratorType(
        position_dims=domain.dims, defined_dims=defined_dims, element_type=element_type
    )


def _type_inference_rule_from_function_type(fun_type: ts.FunctionType):
    def type_rule(*args, **kwargs):
        assert type_info.accepts_args(fun_type, with_args=args, with_kwargs=kwargs)
        return fun_type.returns

    return type_rule


@dataclasses.dataclass
class ITIRTypeInference(eve.NodeTranslator):
    """
    TODO
    """

    offset_provider: common.OffsetProvider
    #: Mapping from a dimension name to the actual dimension instance
    dimensions: dict[str, common.Dimension]
    #: Allow sym refs to symbols that have not been declared. Mostly used in testing.
    allow_undeclared_symbols: bool

    @classmethod
    def apply(
        cls,
        node: T,
        *,
        offset_provider: common.OffsetProvider,
        inplace: bool = False,
        allow_undeclared_symbols: bool = False,
    ) -> T:
        instance = cls(
            offset_provider=offset_provider,
            dimensions=(
                _get_dimensions_from_offset_provider(offset_provider)
                | _get_dimensions_from_types(
                    node.pre_walk_values()
                    .if_isinstance(itir.Node)
                    .getattr("type")
                    .if_is_not(None)
                    .to_list()
                )
            ),
            allow_undeclared_symbols=allow_undeclared_symbols,
        )
        if not inplace:
            node = copy.deepcopy(node)
        instance.visit(
            node,
            ctx={
                name: ObservableTypeInferenceRule(
                    type_rule=rules.type_inference_rules[name],
                    # builtin functions are polymorphic
                    store_inferred_type_in_node=False,
                    offset_provider=offset_provider,
                )
                for name in rules.type_inference_rules.keys()
            },
        )
        return node

    def visit(self, node: concepts.RootNode, **kwargs: Any) -> Any:
        result = super().visit(node, **kwargs)
        if isinstance(node, itir.Node):
            if isinstance(result, ts.TypeSpec):
                if node.type:
                    assert _is_compatible_type(node.type, result)
                node.type = result
            elif isinstance(result, ObservableTypeInferenceRule):
                pass
            elif callable(result):
                return ObservableTypeInferenceRule(
                    node=node,
                    type_rule=result,
                    store_inferred_type_in_node=True,
                    offset_provider=self.offset_provider,
                )
            else:
                raise AssertionError(
                    f"Expected a 'TypeSpec' or 'ObservableTypeInferenceRule', but got "
                    f"`{type(result).__name__}`"
                )
        return result

    def visit_FencilDefinition(self, node: itir.FencilDefinition, *, ctx) -> it_ts.FencilType:
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type

        function_definitions: dict[str, rules.TypeInferenceRule] = {}
        for fun_def in node.function_definitions:
            function_definitions[fun_def.id] = self.visit(fun_def, ctx=ctx | function_definitions)

        closures = self.visit(node.closures, ctx=ctx | params | function_definitions)
        return it_ts.FencilType(params=list(params.values()), closures=closures)

    def visit_FencilWithTemporaries(self, node: global_tmps.FencilWithTemporaries, *, ctx):
        # TODO(tehrengruber): This implementation is not very appealing. Since we are about to
        #  refactor the PR anyway this is fine for now.
        params: dict[str, ts.DataType] = {}
        for param in node.params:
            assert isinstance(param.type, ts.DataType)
            params[param.id] = param.type
        # infer types of temporary declarations
        tmps: dict[str, ts.FieldType] = {}
        for tmp_node in node.tmps:
            tmps[tmp_node.id] = self.visit(tmp_node, ctx=ctx | params)
        # and store them in the inner fencil
        for fencil_param in node.fencil.params:
            if fencil_param.id in tmps:
                fencil_param.type = tmps[fencil_param.id]
        self.visit(node.fencil, ctx=ctx)
        return node.fencil.type

    def visit_Temporary(self, node: itir.Temporary, *, ctx):
        domain = self.visit(node.domain, ctx=ctx)
        assert isinstance(domain, it_ts.DomainType)
        assert node.dtype
        return type_info.apply_to_primitive_constituents(
            lambda dtype: ts.FieldType(dims=domain.dims, dtype=dtype), node.dtype
        )

    def visit_StencilClosure(self, node: itir.StencilClosure, *, ctx) -> it_ts.StencilClosureType:
        domain: it_ts.DomainType = self.visit(node.domain, ctx=ctx)
        inputs: list[ts.FieldType] = self.visit(node.inputs, ctx=ctx)
        output: ts.FieldType = self.visit(node.output, ctx=ctx)

        assert isinstance(domain, it_ts.DomainType)
        for output_el in type_info.primitive_constituents(output):
            assert isinstance(output_el, ts.FieldType)

        stencil_type_rule = self.visit(node.stencil, ctx=ctx)
        stencil_args = [_convert_closure_input_to_iterator(domain, input_) for input_ in inputs]
        stencil_returns = stencil_type_rule(*stencil_args)

        return it_ts.StencilClosureType(
            domain=domain,
            stencil=ts.FunctionType(
                pos_only_args=stencil_args,
                pos_or_kw_args={},
                kw_only_args={},
                returns=stencil_returns,
            ),
            output=output,
            inputs=inputs,
        )

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs) -> ts.DimensionType:
        assert (
            node.value in self.dimensions
        ), f"Dimension {node.value} not present in offset provider."
        return ts.DimensionType(dim=self.dimensions[node.value])

    # TODO: revisit what we want to do with OffsetLiterals as we already have an Offset type in
    #  the frontend.
    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs) -> it_ts.OffsetLiteralType:
        if _is_representable_as_int(node.value):
            return it_ts.OffsetLiteralType(value=int(node.value))
        else:
            assert isinstance(node.value, str) and node.value in self.dimensions
            return it_ts.OffsetLiteralType(value=self.dimensions[node.value])

    def visit_Literal(self, node: itir.Literal, **kwargs) -> ts.ScalarType:
        assert isinstance(node.type, ts.ScalarType)
        return node.type

    def visit_SymRef(
        self, node: itir.SymRef, *, ctx: dict[str, ts.TypeSpec]
    ) -> ts.TypeSpec | rules.TypeInferenceRule:
        # for testing, it is useful to be able to use types without a declaration
        if self.allow_undeclared_symbols and node.id not in ctx:
            # type has been stored in the node itself
            if node.type:
                if isinstance(node.type, ts.FunctionType):
                    return _type_inference_rule_from_function_type(node.type)
                return node.type
            return ts.DeferredType(constraint=None)
        assert node.id in ctx
        result = ctx[node.id]
        if isinstance(result, ObservableTypeInferenceRule):
            result.aliases.append(node)
        return result

    def visit_Lambda(
        self, node: itir.Lambda | itir.FunctionDefinition, *, ctx: dict[str, ts.TypeSpec]
    ) -> ObservableTypeInferenceRule:
        def fun(*args):
            return self.visit(
                node.expr, ctx=ctx | {p.id: a for p, a in zip(node.params, args, strict=True)}
            )

        return ObservableTypeInferenceRule(
            node=node,
            type_rule=fun,
            store_inferred_type_in_node=True,
            offset_provider=self.offset_provider,
        )

    visit_FunctionDefinition = visit_Lambda

    def visit_FunCall(
        self, node: itir.FunCall, *, ctx: dict[str, ts.TypeSpec]
    ) -> ts.TypeSpec | rules.TypeInferenceRule:
        if is_call_to(node, "cast_"):
            value, type_constructor = node.args
            assert (
                isinstance(type_constructor, itir.SymRef)
                and type_constructor.id in itir.TYPEBUILTINS
            )
            return ts.ScalarType(kind=getattr(ts.ScalarKind, type_constructor.id.upper()))

        if is_call_to(node, "tuple_get"):
            index_literal, tuple_ = node.args
            self.visit(tuple_, ctx=ctx)  # ensure tuple is typed
            assert isinstance(index_literal, itir.Literal)
            index = int(index_literal.value)
            assert isinstance(tuple_.type, ts.TupleType)
            return tuple_.type.types[index]

        fun = self.visit(node.fun, ctx=ctx)
        args = self.visit(node.args, ctx=ctx)

        result = fun(*args)

        if isinstance(result, ObservableTypeInferenceRule):
            assert not result.node
            result.node = node

        return result

    def visit_Node(self, node: itir.Node, **kwargs):
        raise NotImplementedError(
            f"No type deduction rule for nodes of type " f"'{type(node).__name__}'."
        )


infer = ITIRTypeInference.apply
