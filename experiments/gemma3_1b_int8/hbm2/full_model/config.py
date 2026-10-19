"""Explicit architecture inputs for the full-model timing simulator."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Architecture:
    core_hz: int = 1_000_000_000
    array_rows: int = 32
    array_cols: int = 32
    operand_staging_bytes: int = 32_768
    vector_staging_bytes: int = 32_768
    vector_lanes: int = 32
    sfu_lanes: int = 4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.operand_staging_bytes < self.array_rows * self.array_cols:
            raise ValueError('Operand staging must hold one complete INT8 array fold')
        for name in ('operand_staging_bytes', 'vector_staging_bytes'):
            if getattr(self, name) % 32:
                raise ValueError(f'{name} must be a multiple of 32 bytes')

    @classmethod
    def read(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    @property
    def k_chunk(self):
        # Split only between complete K folds; a short final fold is permitted.
        return self.array_rows * (self.operand_staging_bytes // (self.array_rows * self.array_cols))

    def to_dict(self):
        return asdict(self)

    def describe(self):
        return dict(
            array=[self.array_rows, self.array_cols],
            processing_elements=self.array_rows * self.array_cols,
            dataflow='WS', vector_lanes=self.vector_lanes, FP32_sfu_lanes=self.sfu_lanes,
            sfu_issue_interval=1,
            sfu_latency_cycles=dict(exp=12, tanh=12, sin_cos=16, rsqrt=16, reciprocal=8),
            descriptor_cycles=4, operand_staging_bytes_each=self.operand_staging_bytes,
            vector_staging_bytes=self.vector_staging_bytes, gemm_k_chunk=self.k_chunk,
            max_inflight_memory_requests=512, max_admissions_per_dram_tick=16,
            weight_layout=f'Prepacked {self.array_cols}-output-column tiles; tied embedding gathered from same storage.',
            scheduling='Serial dependent operators and serial tile load/compute/store; A retained across N tiles; C partials retained across K tiles.',
            traffic='All inter-operator tensors in HBM, including FP32 intermediates, scales, KV updates and layout conversions.',
            compute='Native WS demand counts with staged operands; vector operation passes and reduction/SFU latency are modeled.')
