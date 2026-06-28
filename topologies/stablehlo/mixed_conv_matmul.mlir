// Simple mixed StableHLO MLIR with independent convolutions and matrix multiplication
// For testing mixed workload conversion
module @mixed_conv_matmul {
  func.func public @main(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>,
                         %arg2: tensor<1x64x112x112xf32>, %arg3: tensor<64x64x3x3xf32>,
                         %arg4: tensor<128x256xf32>, %arg5: tensor<256x512xf32>) 
      -> (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<128x512xf32>) {
    
    // Independent Conv 1: 224x224 input, 7x7 filter, 3->64 channels
    %0 = stablehlo.convolution(%arg0, %arg1) 
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], 
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]]} 
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} 
      : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    
    // Independent Conv 2: 112x112 input, 3x3 filter, 64->64 channels
    %1 = stablehlo.convolution(%arg2, %arg3) 
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], 
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} 
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} 
      : (tensor<1x64x112x112xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x112x112xf32>
    
    // Independent MatMul: [128, 256] x [256, 512] -> [128, 512]
    %2 = stablehlo.dot %arg4, %arg5 : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>
    
    return %0, %1, %2 : tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>, tensor<128x512xf32>
  }
}

