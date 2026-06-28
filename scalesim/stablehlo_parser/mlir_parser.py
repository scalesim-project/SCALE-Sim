from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir.ir import Module

from .mlir_common import get_dot_general_dimensions
from .opinfo import OpInfo


class StableHLOParser:
    """Parse a StableHLO MLIR module and extract OpInfo records."""

    _ir_context = None
    _conv_dim_number_re = re.compile(
        r"stablehlo\.conv\s*<\s*(\[\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*\])\s*x\s*"
        r"(\[\s*([io01])\s*,\s*([io01])\s*,\s*([io01])\s*,\s*([io01])\s*\])\s*->\s*"
        r"(\[\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*,\s*([fb01])\s*\])>"
    )

    @staticmethod
    def get_ir_context():
        if StableHLOParser._ir_context is None:
            StableHLOParser._ir_context = make_ir_context()
            # StableHLO printed by JAX often includes dialects that may not be registered
            StableHLOParser._ir_context.allow_unregistered_dialects = True
        return StableHLOParser._ir_context

    @staticmethod
    def parse_module(mlir_text: str, *, context=None) -> Module:
        if context is None:
            context = StableHLOParser.get_ir_context()
        return Module.parse(mlir_text.strip(), context=context)

    def __init__(self, *, mlir_string: Optional[str] = None, mlir_path: Optional[str] = None, mlir_module=None):
        if sum(x is not None for x in [mlir_string, mlir_path, mlir_module]) != 1:
            raise ValueError("Provide exactly one of mlir_string, mlir_path, or mlir_module")

        if mlir_path is not None:
            with open(mlir_path, "r", encoding="utf-8") as f:
                mlir_string = f.read()

        if mlir_string is not None:
            mlir_module = self.parse_module(mlir_string)

        self.module = mlir_module
        self.main_func = self._find_main_func(self.module)

    def _find_main_func(self, module: Module):
        """Best-effort: pick the `main` function; fallback to first callable-looking op.

        Note: With JAX's MLIR bindings, the top-level function may appear with `op.name == "main"`
        (not `func.func`), so we handle both.
        """
        ops = list(module.body.operations)
        for op in ops:
            op_name = str(op.name).strip('"')
            if op_name == "main":
                return op
            if op_name == "func.func":
                try:
                    if "sym_name" in op.attributes and op.attributes["sym_name"].value == "main":
                        return op
                except Exception:
                    # Some attribute maps in the JAX bindings don't fully behave like dicts.
                    pass
        for op in ops:
            op_name = str(op.name).strip('"')
            if op_name in ("func.func", "main"):
                return op
        raise ValueError("No func.func found in module")

    def get_operations(self) -> Iterable[Any]:
        return self.main_func.regions[0].blocks[0].operations

    def _flatten_type(self, t) -> list[Any]:
        """Flatten tuple-like types into a list of element types."""
        # Handle mhlo async bundle by rewriting to tuple for parsing.
        if "!mhlo.async_bundle" in str(t):
            t = t.parse(str(t).replace("!mhlo.async_bundle", "tuple"), context=self.get_ir_context())

        if hasattr(t, "num_types"):
            out: list[Any] = []
            for i in range(t.num_types):
                out.extend(self._flatten_type(t.get_type(i)))
            return out
        return [t]

    def parse_operation(self, op) -> OpInfo:
        name = str(op.name)

        inputs: list[tuple[tuple[int, ...], str]] = []
        outputs: list[tuple[tuple[int, ...], str]] = []

        for operand in op.operands:
            for t in self._flatten_type(operand.type):
                shape = tuple(getattr(t, "shape", ()))
                dtype = str(getattr(t, "element_type", t))
                inputs.append((shape, dtype))

        for result in op.results:
            for t in self._flatten_type(result.type):
                shape = tuple(getattr(t, "shape", ()))
                dtype = str(getattr(t, "element_type", t))
                outputs.append((shape, dtype))

        extra: dict[str, Any] = {}

        if name == "stablehlo.reduce" and "dimensions" in op.attributes:
            extra["dimensions"] = [int(i) for i in op.attributes["dimensions"]]

        elif name == "stablehlo.dot_general":
            lhs_contracting_dims, rhs_contracting_dims, batch_dims = get_dot_general_dimensions(op)
            extra["dims"] = {"lhs": lhs_contracting_dims, "rhs": rhs_contracting_dims, "batch": batch_dims}
            if hasattr(op, "lhs") and hasattr(op, "rhs"):
                extra["lhs_dims"] = list(op.lhs.type.shape)
                extra["rhs_dims"] = list(op.rhs.type.shape)

        elif name == "stablehlo.dot":
            # Basic sanity info (optional)
            if hasattr(op, "lhs") and hasattr(op, "rhs"):
                extra["lhs_dims"] = list(op.lhs.type.shape)
                extra["rhs_dims"] = list(op.rhs.type.shape)

        elif "stablehlo.convolution" in name and hasattr(op, "dimension_numbers"):
            dim_numbers_match = self._conv_dim_number_re.search(str(op.dimension_numbers))
            if dim_numbers_match:
                dim_map = {"lhs": {}, "rhs": {}, "out": {}}
                for dim_type, offset in zip(list(dim_map.keys()), (0, 5, 10)):
                    for element, dim in enumerate(
                        dim_numbers_match.group(2 + offset, 3 + offset, 4 + offset, 5 + offset)
                    ):
                        if dim == "0":
                            dim_map[dim_type]["0_dimension"] = element
                        elif dim == "1":
                            dim_map[dim_type]["1_dimension"] = element
                        elif dim == "f" and dim_type in ["lhs", "out"]:
                            dim_map[dim_type]["feature_dimension"] = element
                        elif dim == "b" and dim_type in ["lhs", "out"]:
                            dim_map[dim_type]["batch_dimension"] = element
                        elif dim == "i" and dim_type in ["rhs"]:
                            dim_map[dim_type]["input_feature_dimension"] = element
                        elif dim == "o" and dim_type in ["rhs"]:
                            dim_map[dim_type]["output_feature_dimension"] = element
                extra["dimension_numbers"] = dim_map

        elif "stablehlo.custom_call" in name:
            if hasattr(op, "call_target_name"):
                extra["kernel_name"] = str(op.call_target_name)

        return OpInfo(name, inputs, outputs, extra=extra)

    def get_ops_list(self) -> list[OpInfo]:
        out: list[OpInfo] = []
        for op in self.get_operations():
            out.append(self.parse_operation(op))
        return out


