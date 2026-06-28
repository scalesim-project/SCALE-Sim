// Simple StableHLO MLIR with matrix multiplications for testing
module @simple_matmul {
  func.func public @main(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512x1024xf32>, %arg3: tensor<1024x10xf32>) -> tensor<128x10xf32> {
    
    // MatMul 1: [128, 256] x [256, 512] -> [128, 512]
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    
    // MatMul 2: [128, 512] x [512, 1024] -> [128, 1024]
    %1 = stablehlo.dot %0, %arg2 : (tensor<128x512xf32>, tensor<512x1024xf32>) -> tensor<128x1024xf32>
    
    // MatMul 3: [128, 1024] x [1024, 10] -> [128, 10]
    %2 = stablehlo.dot %1, %arg3 : (tensor<128x1024xf32>, tensor<1024x10xf32>) -> tensor<128x10xf32>
    
    return %2 : tensor<128x10xf32>
  }
}

