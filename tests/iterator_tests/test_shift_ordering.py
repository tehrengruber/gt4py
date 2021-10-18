from dataclasses import field
import numpy as np
from numpy.core.numeric import allclose
from iterator.runtime import *
from iterator.builtins import *
from iterator.embedded import (
    NeighborTableOffsetProvider,
    np_as_located_field,
    index_field,
)


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")
Cell = CartesianAxis("Cell")

c2e_arr = np.array(
    [
        [0, 1, 2],
    ]
)

e2v_arr = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
    ]
)

C2E = offset("C2E")
E2V = offset("E2V")

@fundef
def v2e2c_stencil(in_vertices):
    return deref(shift(E2V, 1)(shift(C2E, 0)(in_vertices)))


@fendef(offset_provider={"C2E": NeighborTableOffsetProvider(c2e_arr, Cell, Edge, 3),
                         "E2V": NeighborTableOffsetProvider(e2v_arr, Edge, Vertex, 2)})
def v2e2c_fencil(in_vertices, out_cell):
    closure(
        domain(named_range(Cell, 0, 1)),
        v2e2c_stencil,
        [out_cell],
        [in_vertices],
    )

def test_v2e2c():
    inp = np_as_located_field(Vertex)(np.array([666, 42, 666]))
    ref = np.array([42])

    # semantic-model
    out = np_as_located_field(Cell)(np.zeros([1]))

    v2e2c_fencil(inp, out, backend="double_roundtrip")
    assert allclose(out, ref)
