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
    dataflow: str = 'ws'
    balance_k_tiles: bool = False
    gemm_prefetch: bool = False
    fuse_value_cache: bool = False

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in ('dataflow', 'balance_k_tiles', 'gemm_prefetch', 'fuse_value_cache'):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.dataflow not in ('ws', 'os'):
            raise ValueError('dataflow must be ws or os')
        for name in ('balance_k_tiles', 'gemm_prefetch', 'fuse_value_cache'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if self.gemm_prefetch and self.dataflow != 'os':
            raise ValueError('Double-buffered GEMM currently requires OS')
        if self.fuse_value_cache and self.vector_staging_bytes < 8192:
            raise ValueError('Fused V conversion requires at least 8 KiB scratch')
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
        if self.dataflow == 'os':
            return self.operand_staging_bytes // self.array_cols
        # Split only between complete K folds; a short final fold is permitted.
        return self.array_rows * (self.operand_staging_bytes // (self.array_rows * self.array_cols))

    def k_tiles(self, k):
        from math import ceil
        chunk=self.k_chunk
        if self.dataflow=='os' and self.balance_k_tiles:
            chunk=ceil(k/ceil(k/chunk))
        return [(start,min(chunk,k-start)) for start in range(0,k,chunk)]

    def to_dict(self):
        return asdict(self)

    def demand_cycles(self, m, n, k):
        from math import ceil
        if self.dataflow == 'os':
            return ceil(m/self.array_rows)*ceil(n/self.array_cols)*(k+self.array_rows+self.array_cols-2)
        return ceil(k/self.array_rows)*ceil(n/self.array_cols)*(m+2*self.array_rows+self.array_cols-2)

    def describe(self):
        return dict(
            array=[self.array_rows, self.array_cols],
            processing_elements=self.array_rows * self.array_cols,
            dataflow=self.dataflow.upper(), vector_lanes=self.vector_lanes, FP32_sfu_lanes=self.sfu_lanes,
            sfu_issue_interval=1,
            sfu_latency_cycles=dict(exp=12, tanh=12, sin_cos=16, rsqrt=16, reciprocal=8),
            descriptor_cycles=4, operand_staging_bytes_each=self.operand_staging_bytes,
            vector_staging_bytes=self.vector_staging_bytes, gemm_k_chunk=self.k_chunk,
            total_staging_bytes=2*self.operand_staging_bytes+self.vector_staging_bytes,
            operand_banks=('Two ping-pong B banks; A and INT32 partials share vector scratch.' if self.gemm_prefetch else 'Separate A and B banks.'),
            sram_bytes_per_cycle=self.vector_lanes*4,
            operand_bank_read_bytes_per_core_cycle=self.array_cols,
            operand_bank_write_bytes_per_dram_tick=512,
            balanced_k_tiles=self.balance_k_tiles,
            max_inflight_memory_requests=512, max_admissions_per_dram_tick=16,
            weight_layout=f'Prepacked {self.array_cols}-output-column tiles; tied embedding gathered from same storage.',
            scheduling=('Serial operators; callback-gated ping-pong DMA overlaps array computation; A and C partials retained in scratch.' if self.gemm_prefetch else 'Serial dependent operators and serial tile load/compute/store; A retained across N tiles; C partials retained across K tiles.'),
            traffic=('Two streaming passes fuse V dequant/max/requant and remove FP32/layout HBM intermediates; INT8 V output and scales still stored in HBM.' if self.fuse_value_cache else 'All inter-operator tensors in HBM, including FP32 intermediates, scales, KV updates and layout conversions.'),
            compute='Native WS/OS demand counts with staged operands; vector passes, SRAM bandwidth and reduction/SFU latency are modeled.')
