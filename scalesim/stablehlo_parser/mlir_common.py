from __future__ import annotations

import re
from typing import Any


def get_dot_general_dimensions(dot_op: Any) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """Extract batch/contracting dimensions from a stablehlo.dot_general op.

    This mirrors the logic in `hespas_workload.mlir_parser.mlir_common.get_dot_general_dimensions`.
    """
    dot_dims_attr = dot_op.attributes["dot_dimension_numbers"]
    dot_dims_str = str(dot_dims_attr)

    lhs_contracting = re.search(r"lhs_contracting_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    rhs_contracting = re.search(r"rhs_contracting_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    lhs_batch_dims = re.search(r"lhs_batching_dimensions = \[([0-9, ]+)\]", dot_dims_str)
    rhs_batch_dims = re.search(r"rhs_batching_dimensions = \[([0-9, ]+)\]", dot_dims_str)

    lhs_contracting_dims = list(map(int, lhs_contracting.group(1).split(","))) if lhs_contracting else []
    rhs_contracting_dims = list(map(int, rhs_contracting.group(1).split(","))) if rhs_contracting else []
    lhs_batch = list(map(int, lhs_batch_dims.group(1).split(","))) if lhs_batch_dims else []
    rhs_batch = list(map(int, rhs_batch_dims.group(1).split(","))) if rhs_batch_dims else []
    batch_dims = [(lb, rb) for (lb, rb) in zip(lhs_batch, rhs_batch, strict=True)]

    return lhs_contracting_dims, rhs_contracting_dims, batch_dims



