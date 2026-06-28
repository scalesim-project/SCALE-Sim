module {
  func.func @main(%input: tensor<1x3x224x224xf32>, %conv_w: tensor<64x3x7x7xf32>, %fc_w: tensor<4096x1024xf32>) -> tensor<1x1024xf32> {

    // CONV: [1,3,224,224] * [64,3,7,7] -> [1,64,109,109] (stride 2)
    %conv = stablehlo.convolution(%input, %conv_w)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<1x64x109x109xf32>

    // ADD bias (has model)
    %b1 = stablehlo.constant dense<0.1> : tensor<1x64x109x109xf32>
    %conv_bias = stablehlo.add %conv, %b1 : tensor<1x64x109x109xf32>

    // MAXIMUM ReLU (has model)
    %z1 = stablehlo.constant dense<0.0> : tensor<1x64x109x109xf32>
    %relu1 = stablehlo.maximum %conv_bias, %z1 : tensor<1x64x109x109xf32>

    // MULTIPLY scale (has model)
    %s1 = stablehlo.constant dense<0.5> : tensor<1x64x109x109xf32>
    %scaled1 = stablehlo.multiply %relu1, %s1 : tensor<1x64x109x109xf32>

    // SUBTRACT mean (has model)
    %m1 = stablehlo.constant dense<0.25> : tensor<1x64x109x109xf32>
    %centered1 = stablehlo.subtract %scaled1, %m1 : tensor<1x64x109x109xf32>

    // MINIMUM clip (has model)
    %clip1 = stablehlo.constant dense<1.0> : tensor<1x64x109x109xf32>
    %clipped1 = stablehlo.minimum %centered1, %clip1 : tensor<1x64x109x109xf32>

    // ADD offset (has model)
    %off1 = stablehlo.constant dense<0.5> : tensor<1x64x109x109xf32>
    %shifted1 = stablehlo.add %clipped1, %off1 : tensor<1x64x109x109xf32>

    // Reshape flatten
    %flat = stablehlo.reshape %shifted1 : (tensor<1x64x109x109xf32>) -> tensor<1x760384xf32>

    // Slice to 4096 for FC
    %sliced = stablehlo.slice %flat [0:1, 0:4096] : (tensor<1x760384xf32>) -> tensor<1x4096xf32>

    // GEMM: [1,4096] x [4096,1024] -> [1,1024]
    %fc = stablehlo.dot_general %sliced, %fc_w,
      batching_dims = [] x [],
      contracting_dims = [1] x [0]
      : (tensor<1x4096xf32>, tensor<4096x1024xf32>) -> tensor<1x1024xf32>

    // ADD FC bias (has model)
    %b2 = stablehlo.constant dense<0.01> : tensor<1x1024xf32>
    %fc_bias = stablehlo.add %fc, %b2 : tensor<1x1024xf32>

    // MAXIMUM ReLU (has model)
    %z2 = stablehlo.constant dense<0.0> : tensor<1x1024xf32>
    %relu2 = stablehlo.maximum %fc_bias, %z2 : tensor<1x1024xf32>

    // MULTIPLY dropout scale (has model)
    %s2 = stablehlo.constant dense<1.1> : tensor<1x1024xf32>
    %dropout = stablehlo.multiply %relu2, %s2 : tensor<1x1024xf32>

    // SUBTRACT norm mean (has model)
    %m2 = stablehlo.constant dense<0.55> : tensor<1x1024xf32>
    %normed = stablehlo.subtract %dropout, %m2 : tensor<1x1024xf32>

    // MINIMUM final clip (has model)
    %clip2 = stablehlo.constant dense<10.0> : tensor<1x1024xf32>
    %output = stablehlo.minimum %normed, %clip2 : tensor<1x1024xf32>

    return %output : tensor<1x1024xf32>
  }
}
