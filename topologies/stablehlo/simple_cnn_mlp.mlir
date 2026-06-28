module {
  func.func @main(%input: tensor<1x3x32x32xf32>, %conv_w: tensor<16x3x3x3xf32>, %fc_w: tensor<1024x256xf32>, %fc2_w: tensor<256x10xf32>) -> tensor<1x10xf32> {

    // CONV: [1,3,32,32] * [16,3,3,3] -> [1,16,30,30]
    %conv = stablehlo.convolution(%input, %conv_w)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>) -> tensor<1x16x30x30xf32>

    // ADD bias
    %b1 = stablehlo.constant dense<0.1> : tensor<1x16x30x30xf32>
    %conv_bias = stablehlo.add %conv, %b1 : tensor<1x16x30x30xf32>

    // MAXIMUM ReLU
    %z1 = stablehlo.constant dense<0.0> : tensor<1x16x30x30xf32>
    %relu1 = stablehlo.maximum %conv_bias, %z1 : tensor<1x16x30x30xf32>

    // Reshape flatten
    %flat = stablehlo.reshape %relu1 : (tensor<1x16x30x30xf32>) -> tensor<1x14400xf32>

    // Slice to 1024
    %sliced = stablehlo.slice %flat [0:1, 0:1024] : (tensor<1x14400xf32>) -> tensor<1x1024xf32>

    // GEMM 1: [1,1024] x [1024,256] -> [1,256]
    %fc1 = stablehlo.dot_general %sliced, %fc_w,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<1x1024xf32>, tensor<1024x256xf32>) -> tensor<1x256xf32>

    // ADD bias
    %b2 = stablehlo.constant dense<0.01> : tensor<1x256xf32>
    %fc1_bias = stablehlo.add %fc1, %b2 : tensor<1x256xf32>

    // MULTIPLY scale
    %s1 = stablehlo.constant dense<0.5> : tensor<1x256xf32>
    %fc1_scaled = stablehlo.multiply %fc1_bias, %s1 : tensor<1x256xf32>

    // MAXIMUM ReLU
    %z2 = stablehlo.constant dense<0.0> : tensor<1x256xf32>
    %relu2 = stablehlo.maximum %fc1_scaled, %z2 : tensor<1x256xf32>

    // GEMM 2: [1,256] x [256,10] -> [1,10]
    %fc2 = stablehlo.dot_general %relu2, %fc2_w,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<1x256xf32>, tensor<256x10xf32>) -> tensor<1x10xf32>

    // ADD bias
    %b3 = stablehlo.constant dense<0.01> : tensor<1x10xf32>
    %fc2_bias = stablehlo.add %fc2, %b3 : tensor<1x10xf32>

    // SUBTRACT mean
    %m1 = stablehlo.constant dense<0.5> : tensor<1x10xf32>
    %centered = stablehlo.subtract %fc2_bias, %m1 : tensor<1x10xf32>

    // MINIMUM clip
    %clip = stablehlo.constant dense<5.0> : tensor<1x10xf32>
    %output = stablehlo.minimum %centered, %clip : tensor<1x10xf32>

    return %output : tensor<1x10xf32>
  }
}
