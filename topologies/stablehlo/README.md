# StableHLO Test Topologies

Simple StableHLO MLIR files for testing the SCALE-Sim converter.

## Files

### simple_conv.mlir
Three convolution operations:
- Conv1: 224x224 → 112x112, 3→64 channels, 7x7 filter, stride 2
- Conv2: 112x112 → 112x112, 64→64 channels, 3x3 filter, stride 1  
- Conv3: 112x112 → 54x54, 64→128 channels, 3x3 filter, stride 2

### simple_matmul.mlir
Three matrix multiplications using `stablehlo.dot`:
- MatMul1: [128, 256] × [256, 512] → [128, 512]
- MatMul2: [128, 512] × [512, 1024] → [128, 1024]
- MatMul3: [128, 1024] × [1024, 10] → [128, 10]

### simple_dot_general.mlir
Two batched matrix multiplications using `stablehlo.dot_general`:
- Batched MatMul1: [4, 128, 256] × [4, 256, 512] → [4, 128, 512]
- Batched MatMul2: [4, 128, 512] × [4, 512, 1024] → [4, 128, 1024]

## Usage

### Test Convolution
```bash
python3 -m scalesim.scale \
  -t ./topologies/stablehlo/simple_conv.mlir \
  -c ./configs/tpuv4.cfg \
  -p ./results/test_conv
```

### Test Matrix Multiplication
```bash
python3 -m scalesim.scale \
  -t ./topologies/stablehlo/simple_matmul.mlir \
  -c ./configs/tpuv4.cfg \
  -p ./results/test_matmul
```

### Test Dot General (Batched)
```bash
python3 -m scalesim.scale \
  -t ./topologies/stablehlo/simple_dot_general.mlir \
  -c ./configs/tpuv4.cfg \
  -p ./results/test_dot_general
```

## Standalone Conversion

Convert to CSV first to inspect:
```bash
python3 -m scalesim.stablehlo_converter \
  ./topologies/stablehlo/simple_conv.mlir \
  -o simple_conv_topology.csv
```

## Expected Results

### simple_conv.mlir
Should convert to 3 convolution layers in GEMM format (MNK):
- Layer 0: Conv 7x7, 3→64 channels
- Layer 1: Conv 3x3, 64→64 channels
- Layer 2: Conv 3x3, 64→128 channels

### simple_matmul.mlir
Should convert to 3 GEMM operations:
- Layer 0: M=128, K=256, N=512
- Layer 1: M=128, K=512, N=1024
- Layer 2: M=128, K=1024, N=10

### simple_dot_general.mlir
Should convert to 2 batched GEMM operations:
- Layer 0: M=512 (4×128), K=256, N=512
- Layer 1: M=512 (4×128), K=512, N=1024

