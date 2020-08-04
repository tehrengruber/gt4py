import time

import matplotlib.pyplot as plt
import numpy as np

import gt4py as gt
import gt4py.gtscript as gtscript
from gt4py.gtscript import Vertex, Edge
from build import BuildContext, BeginStage

# gridtools4py settings
backend = "debug"  # "debug", "numpy", "gtx86", "gtmc", "gtcuda"
backend_opts = {'verbose': True} if backend.startswith('gt') else {}
dtype = np.float64
origin = (0, 0, 0)
rebuild = True

def zeros(storage_shape, backend, dtype, origin=None, mask=None):
    origin = origin or (0, 0, 0)
    origin = tuple(origin[i] if storage_shape[i] > 2 * origin[i] else 0 for i in range(3))
    domain = tuple(storage_shape[i] - 2 * origin[i] for i in range(3))

    gt_storage = gt.storage.zeros(backend=backend, dtype=dtype, shape=storage_shape, mask=mask, default_origin=origin)
    return gt_storage

#@gtscript.stencil(backend=backend, rebuild=rebuild)
#def scale(
#    mesh: gtscript.Mesh,
#    field: gtscript.Field[[gtscript.Vertex], dtype],
#    scaled_field: gtscript.Field[gtscript.Vertex, dtype]
#):
#    with computation(PARALLEL), location(Vertex) as v:
#        scaled_field[v] = 2*field[v]

#@gtscript.stencil(backend="mesh", rebuild=rebuild)
#def scale(
#    field: gtscript.Field[Vertex, dtype],
#    scaled_field: gtscript.Field[Vertex, dtype]
#):
#    with computation(PARALLEL), interval(...), location(Vertex) as v:
#        #scaled_field[v] = 2*field[v]
#        scaled_field[v] = 2*field[v]

def edge_reduction(
    edge_field: gtscript.Field[Edge, dtype],
    vertex_field: gtscript.Field[Vertex, dtype]
):
    with computation(FORWARD), interval(...), location(Edge) as e:
        edge_field[e] = 0.5*sum(vertex_field[v] for v in vertices(e))

class DummyMesh:
    pass

mesh = DummyMesh()

#scale(mesh, field, scaled_field)


ctx = BuildContext(edge_reduction, backend="mesh", frontend="gtscript")
build = BeginStage(ctx)

ir = build.make_next()
srcs = ir.make_next()

print(f"{ir.ctx}")
print(f"{srcs.ctx['src']}")