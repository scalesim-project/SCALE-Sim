"""StableHLO parser for SCALE-Sim (copied from stablehlo_parse_min)."""

from .mlir_parser import StableHLOParser
from .opinfo import OpInfo

__all__ = ['StableHLOParser', 'OpInfo']

