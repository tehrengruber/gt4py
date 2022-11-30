import dataclasses
from typing import Optional

from eve import NodeTranslator, NodeVisitor
from functional.iterator import ir
from functional.iterator.transforms.remap_symbols import RemapSymbolRefs, RenameSymbols


@dataclasses.dataclass
class CountSymbolRefs(NodeVisitor):
    ref_counts: dict[str, int]

    @classmethod
    def apply(cls, node: ir.Node, symbol_names: list[str]) -> dict[str, int]:
        ref_counts = {name: 0 for name in symbol_names}
        active_refs = set(symbol_names)

        obj = cls(ref_counts=ref_counts)
        obj.visit(node, active_refs=active_refs)

        return ref_counts

    def visit_SymRef(self, node: ir.Node, *, active_refs: set[str]):
        if node.id in active_refs:
            self.ref_counts[node.id] += 1

    def visit_Lambda(self, node: ir.Lambda, *, active_refs: set[str]):
        active_refs = active_refs - set(param.id for param in node.params)

        self.generic_visit(node, active_refs=active_refs)


def inline_lambda(
    node: ir.FunCall,
    opcount_preserving=False,
    force_inline_lift=False,
    eligible_params: Optional[list[bool]] = None,
):
    eligible_params = eligible_params or [True] * len(node.fun.params)

    assert len(eligible_params) == len(node.fun.params) == len(node.args)

    if opcount_preserving:
        ref_counts = CountSymbolRefs.apply(node.fun.expr, [p.id for p in node.fun.params])

        for i, param in enumerate(node.fun.params):
            # TODO(tehrengruber): allow inlining more complicated zero-op expressions like
            #  ignore_shift(...)(it_sym)  # noqa: E800
            if ref_counts[param.id] != 1 and not isinstance(node.args[i], (ir.SymRef, ir.Literal)):
                eligible_params[i] = False

    if force_inline_lift:
        for i, arg in enumerate(node.args):
            if (
                isinstance(arg, ir.FunCall)
                and isinstance(arg.fun, ir.FunCall)
                and isinstance(arg.fun.fun, ir.SymRef)
                and arg.fun.fun.id == "lift"
            ):
                eligible_params[i] = True

    if len(eligible_params) != 0 and not any(eligible_params):
        return node

    refs = set().union(
        *(
            arg.pre_walk_values().if_isinstance(ir.SymRef).getattr("id").to_set()
            for i, arg in enumerate(node.args)
            if eligible_params[i]
        )
    )
    syms = node.fun.expr.pre_walk_values().if_isinstance(ir.Sym).getattr("id").to_set()
    clashes = refs & syms
    expr = node.fun.expr
    if clashes:
        # TODO(tehrengruber): find a better way of generating new symbols
        #  in `name_map` that don't collide with each other. E.g. this
        #  must still work:
        # (lambda arg, arg_: (lambda arg_: ...)(arg))(a, b)  # noqa: E800
        name_map = {}

        def new_name(name):
            while name in refs or name in syms or name in name_map.values():
                name += "_"
            return name

        for sym in clashes:
            name_map[sym] = new_name(sym)

        expr = RenameSymbols().visit(expr, name_map=name_map)

    symbol_map = {
        param.id: arg
        for i, (param, arg) in enumerate(zip(node.fun.params, node.args))
        if eligible_params[i]
    }
    new_expr = RemapSymbolRefs().visit(expr, symbol_map=symbol_map)

    if all(eligible_params):
        return new_expr
    else:
        return ir.FunCall(
            fun=ir.Lambda(
                params=[
                    param
                    for param, eligable in zip(node.fun.params, eligible_params)
                    if not eligable
                ],
                expr=new_expr,
            ),
            args=[arg for arg, eligable in zip(node.args, eligible_params) if not eligable],
        )


@dataclasses.dataclass
class InlineLambdas(NodeTranslator):
    """Inline lambda calls by substituting every argument by its value."""

    opcount_preserving: bool

    force_inline_lift: bool

    @classmethod
    def apply(cls, node: ir.Node, opcount_preserving=False, force_inline_lift=False):
        """
        Inline lambda calls by substituting every arguments by its value.

        Examples:
            `(λ(x) → x)(y)` to `y`
            `(λ(x) → x)(y+y)` to `y+y`
            `(λ(x) → x+x)(y+y)` to `y+y+y+y` if not opcount_preserving
            `(λ(x) → x+x)(y+y)` stays as is if opcount_preserving

        Arguments:
            opcount_preserving: Preserve the number of operations, i.e. only
            inline lambda call if the resulting call has the same number of
            operations.
        """
        return cls(
            opcount_preserving=opcount_preserving,
            force_inline_lift=force_inline_lift,
        ).visit(node)

    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            return inline_lambda(
                node,
                opcount_preserving=self.opcount_preserving,
                force_inline_lift=self.force_inline_lift,
            )

        return node
