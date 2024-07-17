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

from __future__ import annotations

import dataclasses
from typing import Any, Generic

from gt4py._core import definitions as core_defs
from gt4py.next import allocators as next_allocators
from gt4py.next.ffront import (
    foast_to_itir,
    foast_to_past,
    func_to_foast,
    func_to_past,
    past_process_args,
    past_to_itir,
    stages as ffront_stages,
)
from gt4py.next.ffront.past_passes import linters as past_linters
from gt4py.next.iterator import ir as itir
from gt4py.next.otf import stages, workflow
from gt4py.next.program_processors import processor_interface as ppi


@workflow.make_step
def foast_to_foast_closure(
    inp: workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.JITArgs],
) -> ffront_stages.FoastClosure:
    from_fieldop = inp.args.kwargs.pop("from_fieldop")
    debug = inp.args.kwargs.pop("debug", inp.data.debug)
    return ffront_stages.FoastClosure(
        foast_op_def=dataclasses.replace(inp.data, debug=debug),
        args=inp.args.args,
        kwargs=inp.args.kwargs,
        closure_vars={inp.data.foast_node.id: from_fieldop},
    )


@dataclasses.dataclass(frozen=True)
class FieldopTransformWorkflow(workflow.NamedStepSequence):
    """Modular workflow for transformations with access to intermediates."""

    func_to_foast: workflow.Workflow[
        workflow.DataWithArgs[
            ffront_stages.FieldOperatorDefinition | ffront_stages.FoastOperatorDefinition,
            stages.JITArgs,
        ],
        workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.JITArgs],
    ] = dataclasses.field(
        default_factory=lambda: workflow.DataOnlyAdapter(
            func_to_foast.OptionalFuncToFoastFactory(cached=True)
        )
    )
    foast_to_foast_closure: workflow.Workflow[
        workflow.DataWithArgs[ffront_stages.FoastOperatorDefinition, stages.JITArgs],
        ffront_stages.FoastClosure,
    ] = dataclasses.field(default=foast_to_foast_closure, metadata={"takes_args": True})
    foast_to_past_closure: workflow.Workflow[
        ffront_stages.FoastClosure, ffront_stages.PastClosure
    ] = dataclasses.field(
        default_factory=lambda: foast_to_past.FoastToPastClosure(
            foast_to_past=workflow.CachedStep(
                foast_to_past.foast_to_past, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure] = (
        dataclasses.field(default=past_process_args.PastProcessArgs(aot_off=False))
    )
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall] = (
        dataclasses.field(default_factory=past_to_itir.JITPastToItirFactory)
    )

    foast_to_itir: workflow.Workflow[ffront_stages.FoastOperatorDefinition, itir.Expr] = (
        dataclasses.field(
            default_factory=lambda: workflow.CachedStep(
                step=foast_to_itir.foast_to_itir, hash_function=ffront_stages.fingerprint_stage
            )
        )
    )

    @property
    def step_order(self) -> list[str]:
        return [
            "func_to_foast",
            "foast_to_foast_closure",
            "foast_to_past_closure",
            "past_transform_args",
            "past_to_itir",
        ]


DEFAULT_FIELDOP_TRANSFORMS = FieldopTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class ProgramTransformWorkflow(workflow.NamedStepSequenceWithArgs):
    """Modular workflow for transformations with access to intermediates."""

    func_to_past: workflow.SkippableStep[
        ffront_stages.ProgramDefinition | ffront_stages.PastProgramDefinition,
        ffront_stages.PastProgramDefinition,
    ] = dataclasses.field(
        default_factory=lambda: func_to_past.OptionalFuncToPastFactory(cached=True)
    )
    past_lint: workflow.Workflow[
        ffront_stages.PastProgramDefinition, ffront_stages.PastProgramDefinition
    ] = dataclasses.field(default_factory=past_linters.LinterFactory)
    past_to_past_closure: workflow.Workflow[
        workflow.DataWithArgs[ffront_stages.PastProgramDefinition, stages.JITArgs],
        ffront_stages.PastClosure,
    ] = dataclasses.field(
        default=lambda inp: ffront_stages.PastClosure(
            definition=dataclasses.replace(
                inp.data, debug=inp.args.kwargs.pop("debug", inp.data.debug)
            ),
            args=inp.args.args,
            kwargs=inp.args.kwargs,
        ),
        metadata={"takes_args": True},
    )
    past_transform_args: workflow.Workflow[ffront_stages.PastClosure, ffront_stages.PastClosure] = (
        dataclasses.field(default=past_process_args.PastProcessArgs(aot_off=False))
    )
    past_to_itir: workflow.Workflow[ffront_stages.PastClosure, stages.ProgramCall] = (
        dataclasses.field(default_factory=past_to_itir.JITPastToItirFactory)
    )


DEFAULT_PROG_TRANSFORMS = ProgramTransformWorkflow()


@dataclasses.dataclass(frozen=True)
class Backend(Generic[core_defs.DeviceTypeT]):
    executor: ppi.ProgramExecutor
    allocator: next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]
    transforms_fop: workflow.Workflow[
        workflow.DataWithArgs[
            ffront_stages.FieldOperatorDefinition, stages.JITArgs | stages.CompileArgSpec
        ],
        stages.AOTProgram,
    ] = DEFAULT_FIELDOP_TRANSFORMS
    transforms_prog: workflow.Workflow[
        workflow.DataWithArgs[
            ffront_stages.ProgramDefinition, stages.JITArgs | stages.CompileArgSpec
        ],
        stages.AOTProgram,
    ] = DEFAULT_PROG_TRANSFORMS

    def __call__(
        self,
        program: ffront_stages.ProgramDefinition | ffront_stages.FieldOperatorDefinition,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(
            program, (ffront_stages.FieldOperatorDefinition, ffront_stages.FoastOperatorDefinition)
        ):
            _ = kwargs.pop("from_fieldop")
            aot_program = self.transforms_fop(
                workflow.DataWithArgs(program, args=stages.JITArgs(args, kwargs))
            )
        else:
            aot_program = self.transforms_prog(
                workflow.DataWithArgs(program, stages.JITArgs(args, kwargs))
            )
        self.executor(
            aot_program.program, *args, column_axis=aot_program.argspec.column_axis, **kwargs
        )

    @property
    def __name__(self) -> str:
        return getattr(self.executor, "__name__", None) or repr(self)

    @property
    def __gt_allocator__(
        self,
    ) -> next_allocators.FieldBufferAllocatorProtocol[core_defs.DeviceTypeT]:
        return self.allocator
