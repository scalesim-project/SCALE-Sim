// Simple StableHLO MLIR with 3 convolutions for testing
module @simple_conv {
  func.func public @main(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>, %arg2: tensor<64x64x3x3xf32>, %arg3: tensor<128x64x3x3xf32>) -> tensor<1x128x56x56xf32> {
    
    // Conv 1: 224x224 input, 7x7 filter, 3->64 channels, stride 2
    // Output: (224 + 2*3 - 7) / 2 + 1 = 112
    %0 = stablehlo.convolution(%arg0, %arg1) 
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], 
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]]} 
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} 
      : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32>
    
    // Conv 2: 112x112 input, 3x3 filter, 64->64 channels, stride 1
    // Output: (112 + 2*1 - 3) / 1 + 1 = 112
    %1 = stablehlo.convolution(%0, %arg2) 
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], 
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]]} 
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} 
      : (tensor<1x64x112x112xf32>, tensor<64x64x3x3xf32>) -> tensor<1x64x112x112xf32>
    
    // Conv 3: 112x112 input, 3x3 filter, 64->128 channels, stride 2
    // Output: (112 + 2*1 - 3) / 2 + 1 = 56
    %2 = stablehlo.convolution(%1, %arg3) 
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], 
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]]} 
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} 
      : (tensor<1x64x112x112xf32>, tensor<128x64x3x3xf32>) -> tensor<1x128x56x56xf32>
    
    return %2 : tensor<1x128x56x56xf32>
  }
}

