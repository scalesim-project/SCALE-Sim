from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpInfo:
    """Lightweight record describing one StableHLO/MLIR operation."""

    op_name: str
    input_types: list[tuple[tuple[int, ...], str]]  # (shape, dtype_str)
    output_types: list[tuple[tuple[int, ...], str]]  # (shape, dtype_str)
    extra: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        def fmt(sig: tuple[tuple[int, ...], str]) -> str:
            shape, dtype = sig
            shape_str = "x".join(str(d) for d in shape) if shape else ""
            return f"{shape_str}x{dtype}" if shape_str else dtype

        inputs = " ".join(fmt(x) for x in self.input_types)
        outputs = " ".join(fmt(x) for x in self.output_types)
        return f"{self.op_name} ({inputs}) -> ({outputs})"

    def to_jsonable(self) -> dict[str, Any]:
        def pack(sig: tuple[tuple[int, ...], str]) -> dict[str, Any]:
            shape, dtype = sig
            return {"shape": list(shape), "dtype": dtype}

        out: dict[str, Any] = {
            "op_name": self.op_name,
            "inputs": [pack(x) for x in self.input_types],
            "outputs": [pack(x) for x in self.output_types],
        }
        if self.extra:
            out["extra"] = self.extra
        return out

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_jsonable(), indent=indent, sort_keys=True)



