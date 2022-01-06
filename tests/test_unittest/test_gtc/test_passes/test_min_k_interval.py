# flake8: noqa: F841
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

import gt4py.storage
from gt4py import gtscript as gs
from gt4py.backend import from_name
from gt4py.gtscript import PARALLEL, computation, interval, stencil
from gt4py.stencil_builder import StencilBuilder
from gtc.passes.gtir_k_boundary import compute_k_boundary, compute_min_k_size
from gtc.passes.gtir_pipeline import prune_unused_parameters


# populated by test_case decorator
test_data: List[Tuple[Callable, Any]] = []
test_data_k_bounds: List[Tuple[Callable, Any]] = []
test_data_min_k_size: List[Tuple[Callable, Any]] = []


def register_test_case(**kwargs):
    def _wrapper(definition):
        globals()["test_data"].append((definition, kwargs))
        for k, v in kwargs.items():
            globals()["test_data_" + k].append((definition, v))
        return definition

    return _wrapper


# stencils with no extent
@register_test_case(k_bounds=(0, 0), min_k_size=0)
def stencil_no_extent_0(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(...):
        field_a = field_b[0, 0, 0]


@register_test_case(k_bounds=(max(0, -2), 0), min_k_size=2)
def stencil_no_extent_1(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 2):
        field_a = field_b[0, 0, 0]


@register_test_case(k_bounds=(max(-1, -2), 0), min_k_size=2)
def stencil_no_extent_2(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(1, 2):
        field_a = field_b[0, 0, 0]


@register_test_case(k_bounds=(max(max(0, -2), max(-2, -2)), 0), min_k_size=3)
def stencil_no_extent_3(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 2):
        field_a = field_b[0, 0, 0]
    with computation(PARALLEL), interval(2, 3):
        field_a = field_b[0, 0, 0]
    with computation(PARALLEL), interval(3, None):
        field_a = field_b[0, 0, 0]


@register_test_case(k_bounds=(0, max(-1, 0)), min_k_size=1)
def stencil_no_extent_4(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(-1, None):
        field_a = field_b[0, 0, 0]


@register_test_case(k_bounds=(max(0, -1), max(-2, 0)), min_k_size=3)
def stencil_no_extent_5(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 1):
        field_a = field_b[0, 0, 0]
    with computation(PARALLEL), interval(-2, None):
        field_a = field_b[0, 0, 0]


# stencils with extent
@register_test_case(k_bounds=(5, -5), min_k_size=0)
def stencil_with_extent_0(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(...):
        field_a = field_b[0, 0, -5]


@register_test_case(k_bounds=(4, 0), min_k_size=2)
def stencil_with_extent_1(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(1, 2):
        field_a = field_b[0, 0, -5]


@register_test_case(k_bounds=(-6, 0), min_k_size=2)
def stencil_with_extent_2(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(1, 2):
        field_a = field_b[0, 0, 5]


@register_test_case(k_bounds=(3, -3), min_k_size=3)
def stencil_with_extent_3(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 2):
        field_a = field_b[0, 0, -1]
    with computation(PARALLEL), interval(2, 3):
        field_a = field_b[0, 0, -5]
    with computation(PARALLEL), interval(3, None):
        field_a = field_b[0, 0, -3]


@register_test_case(k_bounds=(-5, 5), min_k_size=1)
def stencil_with_extent_4(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, -1):
        field_a = field_b[0, 0, 5]
    with computation(PARALLEL), interval(-1, None):
        field_a = field_b[0, 0, 5]


@register_test_case(k_bounds=(5, -5), min_k_size=3)
def stencil_with_extent_5(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 1):
        field_a = field_b[0, 0, -5]
    with computation(PARALLEL), interval(-2, None):
        field_a = field_b[0, 0, -5]


@register_test_case(k_bounds=(5, 3), min_k_size=2)
def stencil_with_extent_6(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(0, 1):
        field_a = field_b[0, 0, -5] + field_b[0, 0, 3]
    with computation(PARALLEL), interval(-1, None):
        field_a = field_b[0, 0, -5] + field_b[0, 0, 3]


@pytest.mark.parametrize("definition,expected_k_bounds", test_data_k_bounds)
def test_k_bounds(definition, expected_k_bounds):
    builder = StencilBuilder(definition, backend=from_name("debug"))
    k_boundary = compute_k_boundary(builder.gtir_pipeline.full(skip=[prune_unused_parameters]))[
        "field_b"
    ]

    assert expected_k_bounds == k_boundary


@pytest.mark.parametrize("definition,expected_min_k_size", test_data_min_k_size)
def test_min_k_size(definition, expected_min_k_size):
    builder = StencilBuilder(definition, backend=from_name("debug"))
    min_k_size = compute_min_k_size(builder.gtir_pipeline.full(skip=[prune_unused_parameters]))

    assert expected_min_k_size == min_k_size


@pytest.mark.parametrize("definition,expected", test_data)
def test_k_bounds_exec(definition, expected):
    expected_k_bounds, expected_min_k_size = expected["k_bounds"], expected["min_k_size"]

    required_field_size = expected_min_k_size + expected_k_bounds[0] + expected_k_bounds[1]

    if required_field_size > 0:
        backend = "gtc:gt:cpu_ifirst"
        compiled_stencil = stencil(backend, definition)
        field_a = gt4py.storage.zeros(
            backend=backend,
            default_origin=(0, 0, 0),
            shape=(1, 1, expected_min_k_size),
            dtype=np.float64,
        )
        field_b = gt4py.storage.ones(
            backend=backend,
            default_origin=(0, 0, 0),
            shape=(1, 1, required_field_size),
            dtype=np.float64,
        )

        # test with correct domain, origin to low
        with pytest.raises(ValueError, match="Origin for field field_b too small"):
            compiled_stencil(
                field_a,
                field_b,
                domain=(1, 1, expected_min_k_size),
                origin={"field_b": (0, 0, expected_k_bounds[0] - 1)},
            )

        # test with correct domain, correct origin
        compiled_stencil(
            field_a,
            field_b,
            domain=(1, 1, expected_min_k_size),
            origin={"field_b": (0, 0, expected_k_bounds[0])},
        )

        # test with wrong domain, correct origin
        with pytest.raises(ValueError, match="Compute domain too small. Sequential axis"):
            compiled_stencil(
                field_a,
                field_b,
                domain=(1, 1, expected_min_k_size - 1),
                origin={"field_b": (0, 0, expected_k_bounds[0])},
            )


def stencil_with_invalid_temporary_access_start(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(...):
        tmp = field_b[0, 0, 0]
        field_a = tmp[0, 0, -1]


def stencil_with_invalid_temporary_access_end(field_a: gs.Field[float], field_b: gs.Field[float]):
    with computation(PARALLEL), interval(...):
        tmp = field_b[0, 0, 0]
        field_a = tmp[0, 0, 1]


@pytest.mark.parametrize(
    "definition",
    [stencil_with_invalid_temporary_access_start, stencil_with_invalid_temporary_access_end],
)
def test_invalid_temporary_access(definition):
    builder = StencilBuilder(definition, backend=from_name("debug"))
    with pytest.raises(TypeError, match="Invalid access with offset in k to temporary field tmp."):
        k_boundary = compute_k_boundary(builder.gtir_pipeline.full(skip=[prune_unused_parameters]))