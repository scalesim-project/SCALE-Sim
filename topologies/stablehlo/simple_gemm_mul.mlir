module @simple_gemm_mul {
  func.func @main(%arg0: tensor<128x256xbf16>, %arg1: tensor<256x512xbf16>, %arg2: tensor<128x512xbf16>) -> tensor<128x512xbf16> {
    // GEMM: Matrix multiplication (dot_general)
    // Performs: result = arg0 @ arg1
    // Input shapes: [128, 256] @ [256, 512] -> [128, 512]
    // Contracting dimension: inner dimension (256)
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<128x256xbf16>, tensor<256x512xbf16>) -> tensor<128x512xbf16>
    
    // Element-wise multiplication
    // Performs: result = %0 * arg2 (element-wise)
    // Both operands have shape [128, 512]
    %1 = stablehlo.multiply %0, %arg2 : tensor<128x512xbf16>
    
    return %1 : tensor<128x512xbf16>
  }
}

