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

from typing import Any

from gt4py.eve.utils import content_hash
from gt4py.next.otf import languages, stages, workflow
from gt4py.next.program_processors import otf_compile_executor
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn_cpu
from gt4py.next.type_system.type_translation import from_value

CPP_WITH_CUDA = languages.LanguageWithHeaderFilesSettings(
    formatter_key="cpp",
    formatter_style="llvm",
    file_extension="cpp.cu",
    header_extension="hpp",
)


gtfn_gpu: otf_compile_executor.OTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
] = otf_compile_executor.OTFCompileExecutor(
    name="gpu_backend",
    otf_workflow=gtfn_cpu.run_gtfn.otf_workflow.replace(
        translation=gtfn_module.GTFNTranslationStep(
            language_settings=CPP_WITH_CUDA, gtfn_backend=gtfn_module.GTFNBackendKind.GPU
        ),
    ),
)

def compilation_hash(otf_closure: stages.ProgramCall) -> int:
    """Given closure compute a hash uniquely determining if we need to recompile."""
    offset_provider = otf_closure.kwargs["offset_provider"]
    return hash(
        (
            otf_closure.program,
            # As the frontend types contain lists they are not hashable. As a workaround we just
            # use content_hash here.
            content_hash(tuple(from_value(arg) for arg in otf_closure.args)),
            id(offset_provider) if offset_provider else None,
            otf_closure.kwargs.get("column_axis", None),
        )
    )

gtfn_gpu_cached = otf_compile_executor.CachedOTFCompileExecutor[
    languages.Cpp, languages.LanguageWithHeaderFilesSettings, languages.Python, Any
](
    name="gpu_backend_cached",
    otf_workflow=workflow.CachedStep(step=gtfn_gpu.otf_workflow, hash_function=compilation_hash),
)  # todo(ricoh): add API for converting an executor to a cached version of itself and vice versa

