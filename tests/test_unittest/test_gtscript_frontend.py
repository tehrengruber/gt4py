# -*- coding: utf-8 -*-
#
# GT4Py - GridTools4Py - GridTools for Python
#
# Copyright (c) 2014-2019, ETH Zurich
# All rights reserved.
#
# This file is part the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import textwrap
import types

import numpy as np
import pytest

from gt4py import gtscript
from gt4py import definitions as gt_definitions
from gt4py.frontend import gtscript_frontend as gt_frontend
from gt4py import utils as gt_utils

from ..utils import id_version


# ---- Utilities -----
frontend = gt_frontend.GTScriptFrontend


def compile_definition(
    definition_func,
    name: str,
    module: str,
    *,
    externals: dict = None,
    dtypes: dict = None,
    rebuild=False,
    **kwargs,
):
    gtscript._set_arg_dtypes(definition_func, dtypes=dtypes or {})
    build_options = gt_definitions.BuildOptions(
        name=name, module=module, rebuild=rebuild, backend_opts=kwargs, build_info=None
    )

    options_id = gt_utils.shashed_id(build_options)
    stencil_id = frontend.get_stencil_id(
        build_options.qualified_name, definition_func, externals, options_id
    )
    gt_frontend.GTScriptParser(
        definition_func, externals=externals or {}, options=build_options
    ).run()

    return stencil_id


# ---- Tests-----

GLOBAL_BOOL_CONSTANT = True
GLOBAL_CONSTANT = 1.0
GLOBAL_NESTED_CONSTANTS = types.SimpleNamespace(A=100, B=200)
GLOBAL_VERY_NESTED_CONSTANTS = types.SimpleNamespace(nested=types.SimpleNamespace(A=1000, B=2000))


@gtscript.function
def add_external_const(a):
    return a + 10.0 + GLOBAL_CONSTANT


@gtscript.function
def identity(field_in):
    return field_in


class TestInlinedExternals:
    def test_all_legal_combinations(self, id_version):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL

            with computation(PARALLEL), interval(...):
                inout_field = (
                    (
                        inout_field[0, 0, 0]
                        + GLOBAL_CONSTANT
                        + GLOBAL_NESTED_CONSTANTS.A
                        + GLOBAL_VERY_NESTED_CONSTANTS.nested.A
                    )
                    if GLOBAL_BOOL_CONSTANT
                    else 0
                )

        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    def test_missing(self, id_version):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptSymbolError, match=r".*MISSING_CONSTANT.*"):
            compile_definition(definition_func, "test_missing_symbol", module, externals=externals)

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + GLOBAL_NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError, match=r".*GLOBAL_NESTED_CONSTANTS.missing.*"
        ):
            compile_definition(
                definition_func, "test_missing_nested_symbol", module, externals=externals
            )

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, id_version, value_type):
        module = f"TestInlinedExternals_test_module_{id_version}"
        externals = {}

        WRONG_VALUE_CONSTANT = value_type()

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        with pytest.raises(gt_frontend.GTScriptSymbolError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestImportedExternals:
    def test_all_legal_combinations(self, id_version):
        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(
            BOOL_CONSTANT=-1.0,
            CONSTANT=-2.0,
            NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200),
            VERY_NESTED_CONSTANTS=types.SimpleNamespace(
                nested=types.SimpleNamespace(A=-1000, B=-2000)
            ),
        )

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL
            from gt4py.__externals__ import (
                BOOL_CONSTANT,
                CONSTANT,
                NESTED_CONSTANTS,
                VERY_NESTED_CONSTANTS,
            )

            with computation(PARALLEL), interval(...):
                inout_field = (
                    (
                        inout_field[0, 0, 0]
                        + CONSTANT
                        + NESTED_CONSTANTS.A
                        + VERY_NESTED_CONSTANTS.nested.A
                    )
                    if GLOBAL_BOOL_CONSTANT
                    else 0
                )

        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    def test_missing(self, id_version):
        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(CONSTANT=-2.0, NESTED_CONSTANTS=types.SimpleNamespace(A=-100, B=-200))

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL
            from gt4py.__externals__ import MISSING_CONSTANT

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + MISSING_CONSTANT

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*MISSING_CONSTANT.*"):
            compile_definition(definition_func, "test_missing_symbol", module, externals=externals)

        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL
            from gt4py.__externals__ import NESTED_CONSTANTS

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + NESTED_CONSTANTS.missing

        with pytest.raises(
            gt_frontend.GTScriptDefinitionError, match=r".*NESTED_CONSTANTS.missing.*"
        ):
            compile_definition(
                definition_func, "test_missing_nested_symbol", module, externals=externals
            )

    @pytest.mark.parametrize("value_type", [str, dict, list])
    def test_wrong_value(self, id_version, value_type):
        def definition_func(inout_field: gtscript.Field[float]):
            from gt4py.__gtscript__ import computation, interval, PARALLEL
            from gt4py.__externals__ import WRONG_VALUE_CONSTANT

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + WRONG_VALUE_CONSTANT

        module = f"TestImportedExternals_test_module_{id_version}"
        externals = dict(WRONG_VALUE_CONSTANT=value_type())

        with pytest.raises(gt_frontend.GTScriptDefinitionError, match=r".*WRONG_VALUE_CONSTANT.*"):
            compile_definition(definition_func, "test_wrong_value", module, externals=externals)


class TestExternalsWithSubroutines:
    def test_all_legal_combinations(self, id_version):
        @gtscript.function
        def _stage_laplacian_x(dx, phi):
            lap = add_external_const(phi[-1, 0, 0] - 2.0 * phi[0, 0, 0] + phi[1, 0, 0]) / (dx * dx)
            return lap

        @gtscript.function
        def _stage_laplacian_y(dy, phi):
            lap = (phi[0, -1, 0] - 2.0 * phi[0, 0, 0] + phi[0, 1, 0]) / (dy * dy)
            return lap

        @gtscript.function
        def _stage_laplacian(dx, dy, phi):
            from __externals__ import stage_laplacian_x, stage_laplacian_y

            lap_x = stage_laplacian_x(dx=dx, phi=phi)
            lap_y = stage_laplacian_y(dy=dy, phi=phi)
            lap = lap_x[0, 0, 0] + lap_y[0, 0, 0]

            return lap

        def definition_func(
            in_phi: gtscript.Field[np.float64],
            in_gamma: gtscript.Field[np.float64],
            out_phi: gtscript.Field[np.float64],
            out_field: gtscript.Field[np.float64],
            *,
            dx: float,
            dy: float,
        ):
            from gt4py.__gtscript__ import computation, interval, PARALLEL, FORWARD, BACKWARD
            from gt4py.__externals__ import stage_laplacian, stage_laplacian_x, stage_laplacian_y

            with computation(PARALLEL), interval(...):
                lap = stage_laplacian(dx=dx, dy=dy, phi=in_phi) + GLOBAL_CONSTANT
                out_phi = in_gamma[0, 0, 0] * lap[0, 0, 0]

            with computation(PARALLEL), interval(...):
                tmp_out = identity(in_phi)
                out_phi = tmp_out + 1

            with computation(PARALLEL), interval(...):
                tmp_out2 = identity(in_gamma)
                out_field = out_phi + tmp_out2

        module = f"TestExternalsWithSubroutines_test_module_{id_version}"
        externals = {
            "stage_laplacian": _stage_laplacian,
            "stage_laplacian_x": _stage_laplacian_x,
            "stage_laplacian_y": _stage_laplacian_y,
        }
        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )


class TestImports:
    def test_all_legal_combinations(self, id_version):
        def definition_func(inout_field: gtscript.Field[float]):
            from __gtscript__ import computation, interval, PARALLEL, FORWARD, BACKWARD
            from __externals__ import EXTERNAL
            from gt4py.__gtscript__ import computation, interval, PARALLEL, FORWARD, BACKWARD
            from gt4py.__externals__ import EXTERNAL

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0] + EXTERNAL

        module = f"TestImports_test_module_{id_version}"
        externals = dict(EXTERNAL=1.0)
        compile_definition(
            definition_func, "test_all_legal_combinations", module, externals=externals
        )

    @pytest.mark.parametrize(
        "id_case,import_line",
        list(
            enumerate(
                [
                    "import gt4py",
                    "from externals import EXTERNAL",
                    "from gt4py import __gtscript__",
                    "from gt4py import __externals__",
                    "from gt4py.gtscript import computation",
                    "from gt4py.externals import EXTERNAL",
                ]
            )
        ),
    )
    def test_wrong_imports(self, id_case, import_line, id_version):
        module = f"TestImports_test_module_{id_version}"
        externals = {}

        definition_source = textwrap.dedent(
            f"""
        def definition_func(inout_field: gtscript.Field[float]):
            {import_line}

            with computation(PARALLEL), interval(...):
                inout_field = inout_field[0, 0, 0]
"""
        )

        context = dict(gtscript=gtscript)
        exec(definition_source, context)
        definition_func = context["definition_func"]
        definition_func.__exec_source__ = definition_source

        with pytest.raises(gt_frontend.GTScriptSyntaxError):
            compile_definition(
                definition_func, f"test_wrong_imports_{id_case}", module, externals=externals
            )


class TestDTypes:
    @pytest.mark.parametrize(
        "id_case,test_dtype",
        list(enumerate([bool, np.bool, int, np.int32, np.int64, float, np.float32, np.float64])),
    )
    def test_all_legal_dtypes(self, id_case, test_dtype, id_version):
        def definition_func(
            in_field: gtscript.Field[test_dtype],
            out_field: gtscript.Field[test_dtype],
            param: test_dtype,
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"
        compile_definition(definition_func, "test_all_legal_dtypes", module)

        def definition_func(
            in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"], param: "dtype"
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"
        compile_definition(
            definition_func, "test_all_legal_dtypes", module, dtypes={"dtype": test_dtype}
        )

    @pytest.mark.parametrize(
        "id_case,test_dtype", list(enumerate([str, np.uint32, np.uint64, dict, map, bytes]))
    )
    def test_invalid_inlined_dtypes(self, id_case, test_dtype, id_version):
        with pytest.raises(ValueError, match=r".*data type descriptor.*"):

            def definition_func(
                in_field: gtscript.Field[test_dtype],
                out_field: gtscript.Field[test_dtype],
                param: test_dtype,
            ):
                with computation(PARALLEL), interval(...):
                    out_field = in_field + param

    @pytest.mark.parametrize(
        "id_case,test_dtype", list(enumerate([str, np.uint32, np.uint64, dict, map, bytes]))
    )
    def test_invalid_external_dtypes(self, id_case, test_dtype, id_version):
        def definition_func(
            in_field: gtscript.Field["dtype"], out_field: gtscript.Field["dtype"], param: "dtype"
        ):
            with computation(PARALLEL), interval(...):
                out_field = in_field + param

        module = f"TestImports_test_module_{id_version}"

        with pytest.raises(ValueError, match=r".*data type descriptor.*"):
            compile_definition(
                definition_func,
                "test_invalid_external_dtypes",
                module,
                dtypes={"dtype": test_dtype},
            )


class TestAssignmentSyntax:
    def test_ellipsis(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                out_field[...] = in_field

    def test_offset(self):
        @gtscript.stencil(backend="debug")
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            with computation(PARALLEL), interval(...):
                out_field[0, 0, 0] = in_field

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field[0, 0, 1] = in_field

        @gtscript.stencil(backend="debug", externals={"offset": 0})
        def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
            from gt4py.__externals__ import offset

            with computation(PARALLEL), interval(...):
                out_field[0, 0, offset] = in_field

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug", externals={"offset": 1})
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                from gt4py.__externals__ import offset

                with computation(PARALLEL), interval(...):
                    out_field[0, 0, offset] = in_field

    def test_slice(self):

        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field[:, :, :] = in_field

    def test_string(self):
        with pytest.raises(gt_frontend.GTScriptSyntaxError):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    out_field["a_key"] = in_field

    def test_temporary(self):
        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="No subscript allowed in assignment to temporaries",
        ):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    tmp[...] = in_field
                    out_field = tmp

        with pytest.raises(
            gt_frontend.GTScriptSyntaxError,
            match="No subscript allowed in assignment to temporaries",
        ):

            @gtscript.stencil(backend="debug")
            def func(in_field: gtscript.Field[np.float_], out_field: gtscript.Field[np.float_]):
                with computation(PARALLEL), interval(...):
                    tmp[0, 0, 0] = 2 * in_field
                    out_field = tmp
