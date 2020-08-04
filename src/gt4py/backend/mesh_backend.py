from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Mapping, Union, Callable

from gt4py import backend as gt_backend
from gt4py import definitions as gt_definitions
from gt4py.utils import text as gt_text
from gt4py import ir as gt_ir

import eve
import gt_toolchain
from gt_toolchain.unstructured import gtir as eve_gt_ir
from gt_toolchain.unstructured.gtir_to_nir import GtirToNir
from gt_toolchain.unstructured.nir_to_ugpu import NirToUgpu
from gt_toolchain.unstructured.ugpu_codegen import UgpuCodeGenerator

# just copied over from debug backend. not important in the beginning, as the arguments/fields
#  are passed in the driver
def mesh_layout(mask):
    ctr = iter(range(sum(mask)))
    layout = [next(ctr) if m else None for m in mask]
    return tuple(layout)

def mesh_is_compatible_layout(field):
    return sum(field.shape) > 0

def mesh_is_compatible_type(field):
    return isinstance(field, np.ndarray)

class EveDIRConverter(gt_ir.IRNodeVisitor):
    """Transform GT4Py DIR into Eve DIR"""

    def __init__(self, definition_ir):
        self.definition_ir = definition_ir
        self.domain = definition_ir.domain
        self.locations = definition_ir.locations
        self.location_type_stack = [] # a stack of location types. primary location at 0, secondary at 1

    def __call__(self):
        return self.visit(self.definition_ir)

    @staticmethod
    def convert_data_type(data_type : gt_ir.DataType):
        assert data_type.name in [dt.name for dt in gt_toolchain.common.DataType]
        return gt_toolchain.common.DataType[data_type.name]

    @staticmethod
    def convert_location_type(location_type : gt_ir.LocationType):
        return gt_toolchain.common.LocationType[location_type.name]

    @staticmethod
    def convert_iteration_order(iteration_order : gt_ir.IterationOrder):
        return gt_toolchain.common.LoopOrder[iteration_order.name]

    @staticmethod
    def convert_reduction_operator(red_op : gt_ir.ReductionOperator):
        return {
            gt_ir.ReductionOperator.SUM : eve_gt_ir.ReduceOperator.ADD,
            gt_ir.ReductionOperator.PRODUCT : eve_gt_ir.ReduceOperator.MUL,
            gt_ir.ReductionOperator.MIN : eve_gt_ir.ReduceOperator.MIN,
            gt_ir.ReductionOperator.MAX : eve_gt_ir.ReduceOperator.MAX,
        }[red_op]

    def _get_field_location_type(self, field_name : str):
        field = [field for field in self.definition_ir.api_fields if field.name == field_name][0] # todo: ugly. put into method in IR
        assert(len(field.axes) == 1)
        return gt_ir.LocationType[field.axes[0]]

    def visit_ScalarLiteral(self, scalar : gt_ir.ScalarLiteral):
        return eve_gt_ir.Literal(value=str(scalar.value), vtype=self.convert_data_type(scalar.data_type), location_type=self.location_type_stack[-1])

    def visit_BinOpExpr(self, binary_op_expr : gt_ir.BinOpExpr):
        return eve_gt_ir.BinaryOp(op=gt_toolchain.common.BinaryOperator[binary_op_expr.op.name],
                                  left=self.visit(binary_op_expr.lhs),
                                  right=self.visit(binary_op_expr.rhs))

    def visit_NeighborComprehension(self, neighbors : gt_ir.NeighborComprehension):
        # todo: there is no labeling of the neighbors in Eve yet
        return eve_gt_ir.NeighborChain(elements=[self.convert_location_type(link) for link in neighbors.selector.chain])

    def visit_NeighborReduction(self, neighbor_red : gt_ir.NeighborReduction):
        self.location_type_stack.append(neighbor_red.local_field.neighbors.selector.chain[-1])
        operand = self.visit(neighbor_red.local_field.operand)
        self.location_type_stack.pop()

        return eve_gt_ir.NeighborReduce(operand=operand,
                                        op=self.convert_reduction_operator(neighbor_red.op),
                                        neighbors=self.visit(neighbor_red.local_field.neighbors),
                                        location_type=self.location_type_stack[-1])

    def visit_FieldDecl(self, field_decl : gt_ir.FieldDecl):
        # todo: properly transform axes
        assert len(field_decl.axes) == 1
        location_type = self.convert_location_type(gt_ir.LocationType[field_decl.axes[0]])
        dimensions = eve_gt_ir.Dimensions(horizontal=eve_gt_ir.HorizontalDimension(primary=location_type))
        return eve_gt_ir.UField(name=field_decl.name, vtype=self.convert_data_type(field_decl.data_type), dimensions=dimensions)

    def visit_FieldRef(self, field_ref : gt_ir.FieldRef):
        # todo: actually use the "offset"
        # todo: WARNING: the offset is just ignored since it is not supported in Eve. This will result in incorrect code
        #  for reductions over neighbours where the primary and secondary location have same type
        #assert all(offset == 0 for offset in field_ref.offset.values())
        location_type = self.convert_location_type(self._get_field_location_type(field_ref.name))
        return eve_gt_ir.FieldAccess(name=field_ref.name, location_type=location_type)

    def visit_Assign(self, assign_stmt : gt_ir.Assign):
        return eve_gt_ir.AssignStmt(left=self.visit(assign_stmt.target), right=self.visit(assign_stmt.value))

    def visit_ComputationBlock(self, comp_block : gt_ir.ComputationBlock):
        # todo: eve gt_ir does not support interval spec yet
        assert comp_block.interval.start.level == gt_ir.LevelMarker.START
        assert comp_block.interval.end.level == gt_ir.LevelMarker.END

        assert len(self.domain.parallel_axes) == 1 # only one unstructured dimension allowed
        assert self.domain.parallel_axes[0].name in ["Vertex", "Edge", "Cell"]
        # todo: location type is currently attached to the stencil not the computation block
        #  does it even make sense to nest the horizontal loops inside the vertical loops? if the vertical loop
        #  is parallel too, it might make sense make it the inner most loop
        self.location_type_stack.append(self.convert_location_type(gt_ir.LocationType[self.domain.parallel_axes[0].name]))
        horizontal_loops = [eve_gt_ir.HorizontalLoop(location_type=self.location_type_stack[-1], stmt=self.visit(stmt)) for stmt in comp_block.body.stmts]

        result = eve_gt_ir.Stencil(declarations=[], vertical_loops=[
            eve_gt_ir.VerticalLoop(loop_order=self.convert_iteration_order(comp_block.iteration_order),
                                   horizontal_loops=horizontal_loops)])

        self.location_type_stack.pop()

        return result

    def visit_StencilDefinition(self, stencil_def: gt_ir.StencilDefinition):
        # todo: rename params in eve to fields
        assert len(stencil_def.parameters) == 0 # Eve does not support parameters yet

        return eve_gt_ir.Computation(name="DUMMY",  # todo
                                     params=[self.visit(field) for field in stencil_def.api_fields],
                                     stencils=[self.visit(stencil) for stencil in stencil_def.computations])

@gt_backend.register
class MeshBackend(gt_backend.BaseBackend, gt_backend.PurePythonBackendCLIMixin):
    """Prototyping backend for GT4Py on Meshes."""

    name = "mesh"
    options = {}
    storage_info = {
        "alignment": 1,
        "device": "cpu",
        "layout_map": mesh_layout,
        "is_compatible_layout": mesh_is_compatible_layout,
        "is_compatible_type": mesh_is_compatible_type,
    }
    languages = {"computation": "python", "bindings": []}

    MODULE_GENERATOR_CLASS = None

    def _naive_file_name(build_options: gt_definitions.BuildOptions) -> str:
        return build_options.name + ".py"

    @classmethod
    def _generate_module_source(cls,
        definition_ir: gt_ir.StencilDefinition,
        options: gt_definitions.BuildOptions,
        *,
        stencil_id: Optional[gt_definitions.StencilID] = None,
        **kwargs,
    ):
        comp = EveDIRConverter(definition_ir)()

        nir_comp = GtirToNir().visit(comp)
        # debug(nir_comp)
        ugpu_comp = NirToUgpu().visit(nir_comp)
        # debug(ugpu_comp)

        generated_code = UgpuCodeGenerator.apply(ugpu_comp)
        return generated_code