# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Optional

from gt4py.eve import utils as eve_utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.transforms import (
    fuse_as_fieldop,
    global_tmps,
    infer_domain,
    inline_fundefs,
)
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.type_system.inference import infer


# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `extract_temporaries` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Node,
    *,
    extract_temporaries=False,
    offset_provider=None,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    unconditionally_collapse_tuples=False,
    # FIXME[#1582](tehrengruber): Revisit and cleanup after new GTIR temporary pass is in place
    temporary_extraction_heuristics: Optional[
        Callable[[itir.StencilClosure], Callable[[itir.Expr], bool]]
    ] = None,
    symbolic_domain_sizes: Optional[dict[str, str]] = None,
) -> itir.Program:
    assert isinstance(ir, itir.Program)

    tmp_uids = eve_utils.UIDGenerator(prefix="__tmp")
    mergeasfop_uids = eve_utils.UIDGenerator()

    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)  # type: ignore[arg-type] # all previous passes return itir.Program
    ir = NormalizeShifts().visit(ir)

    # note: this increases the size of the tree
    # Inline. The domain inference can not handle "user" functions, e.g. `let f = λ(...) → ... in f(...)`
    ir = InlineLambdas.apply(ir, opcount_preserving=True, force_inline_lambda_args=True)
    ir = infer_domain.infer_program(
        ir,  # type: ignore[arg-type]  # always an itir.Program
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
    )

    for _ in range(10):
        inlined = ir

        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]  # always an itir.Program
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(inlined, offset_provider=offset_provider)  # type: ignore[assignment]  # always an itir.Program

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
            inlined, uids=mergeasfop_uids, offset_provider=offset_provider
        )

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(ir, offset_provider=offset_provider)
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider=offset_provider)
        ir = global_tmps.create_global_tmps(ir, offset_provider=offset_provider, uids=tmp_uids)  # type: ignore[arg-type]  # always an itir.Program

    # Since `CollapseTuple` relies on the type inference which does not support returning tuples
    # larger than the number of closure outputs as given by the unconditional collapse, we can
    # only run the unconditional version here instead of in the loop above.
    if unconditionally_collapse_tuples:
        ir = CollapseTuple.apply(ir, ignore_tuple_size=True, offset_provider=offset_provider)

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps().visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider=offset_provider)
            if unrolled == ir:
                break
            ir = unrolled
            ir = CollapseListGet().visit(ir)
            ir = NormalizeShifts().visit(ir)
        else:
            raise RuntimeError("Reduction unrolling failed.")

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir
