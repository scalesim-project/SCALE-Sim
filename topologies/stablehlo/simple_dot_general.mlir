// Simple StableHLO MLIR with dot_general (batched matmul) for testing
module @simple_dot_general {
  func.func public @main(%arg0: tensor<4x128x256xf32>, %arg1: tensor<4x256x512xf32>, %arg2: tensor<4x512x1024xf32>) -> tensor<4x128x1024xf32> {
    
    // Batched MatMul 1: [4, 128, 256] x [4, 256, 512] -> [4, 128, 512]
    %0 = stablehlo.dot_general %arg0, %arg1, 
      batching_dims = [0] x [0], 
      contracting_dims = [2] x [1] 
      : (tensor<4x128x256xf32>, tensor<4x256x512xf32>) -> tensor<4x128x512xf32>
    
    // Batched MatMul 2: [4, 128, 512] x [4, 512, 1024] -> [4, 128, 1024]
    %1 = stablehlo.dot_general %0, %arg2, 
      batching_dims = [0] x [0], 
      contracting_dims = [2] x [1] 
      : (tensor<4x128x512xf32>, tensor<4x512x1024xf32>) -> tensor<4x128x1024xf32>
    
    return %1 : tensor<4x128x1024xf32>
  }
}

