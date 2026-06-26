module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<768xf32>, %arg1: tensor<512x768xf32>, %arg2: tensor<512xf32>, %arg3: tensor<768x512xf32>, %arg4: tensor<512xf32>, %arg5: tensor<512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<512xf32>, %arg9: tensor<512x512xf32>, %arg10: tensor<1536xf32>, %arg11: tensor<512x1536xf32>, %arg12: tensor<768xf32>, %arg13: tensor<512x768xf32>, %arg14: tensor<512xf32>, %arg15: tensor<768x512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512xf32>, %arg18: tensor<512xf32>, %arg19: tensor<512xf32>, %arg20: tensor<512xf32>, %arg21: tensor<512x512xf32>, %arg22: tensor<1536xf32>, %arg23: tensor<512x1536xf32>, %arg24: tensor<768xf32>, %arg25: tensor<512x768xf32>, %arg26: tensor<512xf32>, %arg27: tensor<768x512xf32>, %arg28: tensor<512xf32>, %arg29: tensor<512xf32>, %arg30: tensor<512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<512xf32>, %arg33: tensor<512x512xf32>, %arg34: tensor<1536xf32>, %arg35: tensor<512x1536xf32>, %arg36: tensor<2048x512xf32>, %arg37: tensor<512x2048xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<1x128xi32>) -> (tensor<1x128x2048xf32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x128xi32>
    %1 = stablehlo.compare  LT, %arg40, %0,  SIGNED : (tensor<1x128xi32>, tensor<1x128xi32>) -> tensor<1x128xi1>
    %c_0 = stablehlo.constant dense<2048> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x128xi32>
    %3 = stablehlo.add %arg40, %2 : tensor<1x128xi32>
    %4 = stablehlo.select %1, %3, %arg40 : tensor<1x128xi1>, tensor<1x128xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<1x128x1xi32>
    %6 = "stablehlo.gather"(%arg36, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<2048x512xf32>, tensor<1x128x1xi32>) -> tensor<1x128x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %cst_1 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<1x128x1xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %12 = stablehlo.subtract %6, %11 : tensor<1x128x512xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<1x128x512xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %14 = stablehlo.reduce(%13 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %15 = stablehlo.broadcast_in_dim %14, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %16 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %17 = stablehlo.divide %15, %16 : tensor<1x128x1xf32>
    %18 = stablehlo.broadcast_in_dim %10, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %19 = stablehlo.subtract %6, %18 : tensor<1x128x512xf32>
    %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %21 = stablehlo.add %17, %20 : tensor<1x128x1xf32>
    %22 = stablehlo.rsqrt %21 : tensor<1x128x1xf32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %24 = stablehlo.multiply %19, %23 : tensor<1x128x512xf32>
    %25 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %27 = stablehlo.multiply %24, %26 : tensor<1x128x512xf32>
    %28 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %30 = stablehlo.add %27, %29 : tensor<1x128x512xf32>
    %31 = stablehlo.dot_general %30, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x1536xf32>) -> tensor<1x128x1536xf32>
    %32 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x128x1536xf32>
    %34 = stablehlo.add %31, %33 : tensor<1x128x1536xf32>
    %35 = stablehlo.reshape %34 : (tensor<1x128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %36 = stablehlo.slice %35 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %37 = stablehlo.reshape %36 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %38 = stablehlo.transpose %37, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %39 = stablehlo.slice %35 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %40 = stablehlo.reshape %39 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %41 = stablehlo.transpose %40, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %42 = stablehlo.slice %35 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %43 = stablehlo.reshape %42 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %44 = stablehlo.transpose %43, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %45 = stablehlo.transpose %41, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %46 = stablehlo.reshape %38 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %47 = stablehlo.dot_general %46, %45, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %48 = stablehlo.transpose %47, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %cst_4 = stablehlo.constant dense<0.0883883461> : tensor<f32>
    %49 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %50 = stablehlo.multiply %48, %49 : tensor<1x4x128x128xf32>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %52 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4x128xf32>
    %53 = stablehlo.maximum %52, %51 : tensor<1x4x128xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %55 = stablehlo.broadcast_in_dim %54, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %56 = stablehlo.subtract %50, %55 : tensor<1x4x128x128xf32>
    %57 = stablehlo.exponential %56 : tensor<1x4x128x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %58 = stablehlo.reduce(%57 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %60 = stablehlo.broadcast_in_dim %59, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %61 = stablehlo.divide %57, %60 : tensor<1x4x128x128xf32>
    %62 = stablehlo.reshape %61 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %63 = stablehlo.dot_general %62, %44, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %64 = stablehlo.transpose %63, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %65 = stablehlo.transpose %64, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %66 = stablehlo.reshape %65 : (tensor<1x128x4x128xf32>) -> tensor<1x128x512xf32>
    %67 = stablehlo.dot_general %66, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x512xf32>) -> tensor<1x128x512xf32>
    %68 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %70 = stablehlo.add %67, %69 : tensor<1x128x512xf32>
    %71 = stablehlo.add %6, %70 : tensor<1x128x512xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %72 = stablehlo.reduce(%71 init: %cst_8) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %73 = stablehlo.broadcast_in_dim %72, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %74 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %75 = stablehlo.divide %73, %74 : tensor<1x128x1xf32>
    %76 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %77 = stablehlo.subtract %71, %76 : tensor<1x128x512xf32>
    %78 = stablehlo.multiply %77, %77 : tensor<1x128x512xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %81 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %82 = stablehlo.divide %80, %81 : tensor<1x128x1xf32>
    %83 = stablehlo.broadcast_in_dim %75, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %84 = stablehlo.subtract %71, %83 : tensor<1x128x512xf32>
    %85 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %86 = stablehlo.add %82, %85 : tensor<1x128x1xf32>
    %87 = stablehlo.rsqrt %86 : tensor<1x128x1xf32>
    %88 = stablehlo.broadcast_in_dim %87, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %89 = stablehlo.multiply %84, %88 : tensor<1x128x512xf32>
    %90 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %92 = stablehlo.multiply %89, %91 : tensor<1x128x512xf32>
    %93 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %94 = stablehlo.broadcast_in_dim %93, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %95 = stablehlo.add %92, %94 : tensor<1x128x512xf32>
    %96 = stablehlo.dot_general %95, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x768xf32>) -> tensor<1x128x768xf32>
    %97 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x128x768xf32>
    %99 = stablehlo.add %96, %98 : tensor<1x128x768xf32>
    %100 = stablehlo.multiply %99, %99 : tensor<1x128x768xf32>
    %101 = stablehlo.multiply %100, %99 : tensor<1x128x768xf32>
    %cst_10 = stablehlo.constant dense<4.471500e-02> : tensor<f32>
    %102 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %103 = stablehlo.multiply %102, %101 : tensor<1x128x768xf32>
    %104 = stablehlo.add %99, %103 : tensor<1x128x768xf32>
    %cst_11 = stablehlo.constant dense<0.797884583> : tensor<f32>
    %105 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %106 = stablehlo.multiply %105, %104 : tensor<1x128x768xf32>
    %107 = stablehlo.tanh %106 : tensor<1x128x768xf32>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %108 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %109 = stablehlo.add %108, %107 : tensor<1x128x768xf32>
    %cst_13 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %110 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %111 = stablehlo.multiply %110, %109 : tensor<1x128x768xf32>
    %112 = stablehlo.multiply %99, %111 : tensor<1x128x768xf32>
    %113 = stablehlo.dot_general %112, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf32>, tensor<768x512xf32>) -> tensor<1x128x512xf32>
    %114 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %115 = stablehlo.broadcast_in_dim %114, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %116 = stablehlo.add %113, %115 : tensor<1x128x512xf32>
    %117 = stablehlo.add %71, %116 : tensor<1x128x512xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %118 = stablehlo.reduce(%117 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %119 = stablehlo.broadcast_in_dim %118, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %120 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %121 = stablehlo.divide %119, %120 : tensor<1x128x1xf32>
    %122 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %123 = stablehlo.subtract %117, %122 : tensor<1x128x512xf32>
    %124 = stablehlo.multiply %123, %123 : tensor<1x128x512xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %125 = stablehlo.reduce(%124 init: %cst_15) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %126 = stablehlo.broadcast_in_dim %125, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %127 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %128 = stablehlo.divide %126, %127 : tensor<1x128x1xf32>
    %129 = stablehlo.broadcast_in_dim %121, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %130 = stablehlo.subtract %117, %129 : tensor<1x128x512xf32>
    %131 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %132 = stablehlo.add %128, %131 : tensor<1x128x1xf32>
    %133 = stablehlo.rsqrt %132 : tensor<1x128x1xf32>
    %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %135 = stablehlo.multiply %130, %134 : tensor<1x128x512xf32>
    %136 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %137 = stablehlo.broadcast_in_dim %136, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %138 = stablehlo.multiply %135, %137 : tensor<1x128x512xf32>
    %139 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %140 = stablehlo.broadcast_in_dim %139, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %141 = stablehlo.add %138, %140 : tensor<1x128x512xf32>
    %142 = stablehlo.dot_general %141, %arg23, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x1536xf32>) -> tensor<1x128x1536xf32>
    %143 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %144 = stablehlo.broadcast_in_dim %143, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x128x1536xf32>
    %145 = stablehlo.add %142, %144 : tensor<1x128x1536xf32>
    %146 = stablehlo.reshape %145 : (tensor<1x128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %147 = stablehlo.slice %146 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %148 = stablehlo.reshape %147 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %149 = stablehlo.transpose %148, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %150 = stablehlo.slice %146 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %151 = stablehlo.reshape %150 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %152 = stablehlo.transpose %151, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %153 = stablehlo.slice %146 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %154 = stablehlo.reshape %153 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %155 = stablehlo.transpose %154, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %156 = stablehlo.transpose %152, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %157 = stablehlo.reshape %149 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %158 = stablehlo.dot_general %157, %156, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %159 = stablehlo.transpose %158, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %160 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %161 = stablehlo.multiply %159, %160 : tensor<1x4x128x128xf32>
    %cst_16 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %162 = stablehlo.reduce(%161 init: %cst_16) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %163 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4x128xf32>
    %164 = stablehlo.maximum %163, %162 : tensor<1x4x128xf32>
    %165 = stablehlo.broadcast_in_dim %164, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %166 = stablehlo.broadcast_in_dim %165, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %167 = stablehlo.subtract %161, %166 : tensor<1x4x128x128xf32>
    %168 = stablehlo.exponential %167 : tensor<1x4x128x128xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %169 = stablehlo.reduce(%168 init: %cst_17) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %170 = stablehlo.broadcast_in_dim %169, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %171 = stablehlo.broadcast_in_dim %170, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %172 = stablehlo.divide %168, %171 : tensor<1x4x128x128xf32>
    %173 = stablehlo.reshape %172 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %174 = stablehlo.dot_general %173, %155, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %175 = stablehlo.transpose %174, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %176 = stablehlo.transpose %175, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %177 = stablehlo.reshape %176 : (tensor<1x128x4x128xf32>) -> tensor<1x128x512xf32>
    %178 = stablehlo.dot_general %177, %arg21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x512xf32>) -> tensor<1x128x512xf32>
    %179 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %181 = stablehlo.add %178, %180 : tensor<1x128x512xf32>
    %182 = stablehlo.add %117, %181 : tensor<1x128x512xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %183 = stablehlo.reduce(%182 init: %cst_18) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %184 = stablehlo.broadcast_in_dim %183, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %185 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %186 = stablehlo.divide %184, %185 : tensor<1x128x1xf32>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %188 = stablehlo.subtract %182, %187 : tensor<1x128x512xf32>
    %189 = stablehlo.multiply %188, %188 : tensor<1x128x512xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %190 = stablehlo.reduce(%189 init: %cst_19) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %191 = stablehlo.broadcast_in_dim %190, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %192 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %193 = stablehlo.divide %191, %192 : tensor<1x128x1xf32>
    %194 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %195 = stablehlo.subtract %182, %194 : tensor<1x128x512xf32>
    %196 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %197 = stablehlo.add %193, %196 : tensor<1x128x1xf32>
    %198 = stablehlo.rsqrt %197 : tensor<1x128x1xf32>
    %199 = stablehlo.broadcast_in_dim %198, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %200 = stablehlo.multiply %195, %199 : tensor<1x128x512xf32>
    %201 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %202 = stablehlo.broadcast_in_dim %201, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %203 = stablehlo.multiply %200, %202 : tensor<1x128x512xf32>
    %204 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %206 = stablehlo.add %203, %205 : tensor<1x128x512xf32>
    %207 = stablehlo.dot_general %206, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x768xf32>) -> tensor<1x128x768xf32>
    %208 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %209 = stablehlo.broadcast_in_dim %208, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x128x768xf32>
    %210 = stablehlo.add %207, %209 : tensor<1x128x768xf32>
    %211 = stablehlo.multiply %210, %210 : tensor<1x128x768xf32>
    %212 = stablehlo.multiply %211, %210 : tensor<1x128x768xf32>
    %213 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %214 = stablehlo.multiply %213, %212 : tensor<1x128x768xf32>
    %215 = stablehlo.add %210, %214 : tensor<1x128x768xf32>
    %216 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %217 = stablehlo.multiply %216, %215 : tensor<1x128x768xf32>
    %218 = stablehlo.tanh %217 : tensor<1x128x768xf32>
    %219 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %220 = stablehlo.add %219, %218 : tensor<1x128x768xf32>
    %221 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %222 = stablehlo.multiply %221, %220 : tensor<1x128x768xf32>
    %223 = stablehlo.multiply %210, %222 : tensor<1x128x768xf32>
    %224 = stablehlo.dot_general %223, %arg15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf32>, tensor<768x512xf32>) -> tensor<1x128x512xf32>
    %225 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %226 = stablehlo.broadcast_in_dim %225, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %227 = stablehlo.add %224, %226 : tensor<1x128x512xf32>
    %228 = stablehlo.add %182, %227 : tensor<1x128x512xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %229 = stablehlo.reduce(%228 init: %cst_20) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %230 = stablehlo.broadcast_in_dim %229, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %231 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %232 = stablehlo.divide %230, %231 : tensor<1x128x1xf32>
    %233 = stablehlo.broadcast_in_dim %232, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %234 = stablehlo.subtract %228, %233 : tensor<1x128x512xf32>
    %235 = stablehlo.multiply %234, %234 : tensor<1x128x512xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %236 = stablehlo.reduce(%235 init: %cst_21) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %237 = stablehlo.broadcast_in_dim %236, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %238 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %239 = stablehlo.divide %237, %238 : tensor<1x128x1xf32>
    %240 = stablehlo.broadcast_in_dim %232, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %241 = stablehlo.subtract %228, %240 : tensor<1x128x512xf32>
    %242 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %243 = stablehlo.add %239, %242 : tensor<1x128x1xf32>
    %244 = stablehlo.rsqrt %243 : tensor<1x128x1xf32>
    %245 = stablehlo.broadcast_in_dim %244, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %246 = stablehlo.multiply %241, %245 : tensor<1x128x512xf32>
    %247 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %248 = stablehlo.broadcast_in_dim %247, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %249 = stablehlo.multiply %246, %248 : tensor<1x128x512xf32>
    %250 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %252 = stablehlo.add %249, %251 : tensor<1x128x512xf32>
    %253 = stablehlo.dot_general %252, %arg35, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x1536xf32>) -> tensor<1x128x1536xf32>
    %254 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<1536xf32>) -> tensor<1x1x1536xf32>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x1x1536xf32>) -> tensor<1x128x1536xf32>
    %256 = stablehlo.add %253, %255 : tensor<1x128x1536xf32>
    %257 = stablehlo.reshape %256 : (tensor<1x128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %258 = stablehlo.slice %257 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %259 = stablehlo.reshape %258 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %260 = stablehlo.transpose %259, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %261 = stablehlo.slice %257 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %262 = stablehlo.reshape %261 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %263 = stablehlo.transpose %262, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %264 = stablehlo.slice %257 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf32>) -> tensor<1x128x1x4x128xf32>
    %265 = stablehlo.reshape %264 : (tensor<1x128x1x4x128xf32>) -> tensor<1x128x4x128xf32>
    %266 = stablehlo.transpose %265, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf32>) -> tensor<1x4x128x128xf32>
    %267 = stablehlo.transpose %263, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %268 = stablehlo.reshape %260 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %269 = stablehlo.dot_general %268, %267, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %270 = stablehlo.transpose %269, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %271 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f32>) -> tensor<1x4x128x128xf32>
    %272 = stablehlo.multiply %270, %271 : tensor<1x4x128x128xf32>
    %cst_22 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %273 = stablehlo.reduce(%272 init: %cst_22) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %274 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f32>) -> tensor<1x4x128xf32>
    %275 = stablehlo.maximum %274, %273 : tensor<1x4x128xf32>
    %276 = stablehlo.broadcast_in_dim %275, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %277 = stablehlo.broadcast_in_dim %276, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %278 = stablehlo.subtract %272, %277 : tensor<1x4x128x128xf32>
    %279 = stablehlo.exponential %278 : tensor<1x4x128x128xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %280 = stablehlo.reduce(%279 init: %cst_23) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %281 = stablehlo.broadcast_in_dim %280, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %282 = stablehlo.broadcast_in_dim %281, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x128xf32>
    %283 = stablehlo.divide %279, %282 : tensor<1x4x128x128xf32>
    %284 = stablehlo.reshape %283 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %285 = stablehlo.dot_general %284, %266, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<1x4x128x128xf32>) -> tensor<4x128x1x128xf32>
    %286 = stablehlo.transpose %285, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf32>) -> tensor<1x4x128x128xf32>
    %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %288 = stablehlo.reshape %287 : (tensor<1x128x4x128xf32>) -> tensor<1x128x512xf32>
    %289 = stablehlo.dot_general %288, %arg33, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x512xf32>) -> tensor<1x128x512xf32>
    %290 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %291 = stablehlo.broadcast_in_dim %290, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %292 = stablehlo.add %289, %291 : tensor<1x128x512xf32>
    %293 = stablehlo.add %228, %292 : tensor<1x128x512xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %294 = stablehlo.reduce(%293 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %295 = stablehlo.broadcast_in_dim %294, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %296 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %297 = stablehlo.divide %295, %296 : tensor<1x128x1xf32>
    %298 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %299 = stablehlo.subtract %293, %298 : tensor<1x128x512xf32>
    %300 = stablehlo.multiply %299, %299 : tensor<1x128x512xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %301 = stablehlo.reduce(%300 init: %cst_25) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %302 = stablehlo.broadcast_in_dim %301, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %303 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %304 = stablehlo.divide %302, %303 : tensor<1x128x1xf32>
    %305 = stablehlo.broadcast_in_dim %297, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %306 = stablehlo.subtract %293, %305 : tensor<1x128x512xf32>
    %307 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %308 = stablehlo.add %304, %307 : tensor<1x128x1xf32>
    %309 = stablehlo.rsqrt %308 : tensor<1x128x1xf32>
    %310 = stablehlo.broadcast_in_dim %309, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %311 = stablehlo.multiply %306, %310 : tensor<1x128x512xf32>
    %312 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %313 = stablehlo.broadcast_in_dim %312, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %314 = stablehlo.multiply %311, %313 : tensor<1x128x512xf32>
    %315 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %316 = stablehlo.broadcast_in_dim %315, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %317 = stablehlo.add %314, %316 : tensor<1x128x512xf32>
    %318 = stablehlo.dot_general %317, %arg25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x768xf32>) -> tensor<1x128x768xf32>
    %319 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<768xf32>) -> tensor<1x1x768xf32>
    %320 = stablehlo.broadcast_in_dim %319, dims = [0, 1, 2] : (tensor<1x1x768xf32>) -> tensor<1x128x768xf32>
    %321 = stablehlo.add %318, %320 : tensor<1x128x768xf32>
    %322 = stablehlo.multiply %321, %321 : tensor<1x128x768xf32>
    %323 = stablehlo.multiply %322, %321 : tensor<1x128x768xf32>
    %324 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %325 = stablehlo.multiply %324, %323 : tensor<1x128x768xf32>
    %326 = stablehlo.add %321, %325 : tensor<1x128x768xf32>
    %327 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %328 = stablehlo.multiply %327, %326 : tensor<1x128x768xf32>
    %329 = stablehlo.tanh %328 : tensor<1x128x768xf32>
    %330 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %331 = stablehlo.add %330, %329 : tensor<1x128x768xf32>
    %332 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<1x128x768xf32>
    %333 = stablehlo.multiply %332, %331 : tensor<1x128x768xf32>
    %334 = stablehlo.multiply %321, %333 : tensor<1x128x768xf32>
    %335 = stablehlo.dot_general %334, %arg27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf32>, tensor<768x512xf32>) -> tensor<1x128x512xf32>
    %336 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %337 = stablehlo.broadcast_in_dim %336, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %338 = stablehlo.add %335, %337 : tensor<1x128x512xf32>
    %339 = stablehlo.add %293, %338 : tensor<1x128x512xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %340 = stablehlo.reduce(%339 init: %cst_26) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %341 = stablehlo.broadcast_in_dim %340, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %342 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %343 = stablehlo.divide %341, %342 : tensor<1x128x1xf32>
    %344 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %345 = stablehlo.subtract %339, %344 : tensor<1x128x512xf32>
    %346 = stablehlo.multiply %345, %345 : tensor<1x128x512xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %347 = stablehlo.reduce(%346 init: %cst_27) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %348 = stablehlo.broadcast_in_dim %347, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %349 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %350 = stablehlo.divide %348, %349 : tensor<1x128x1xf32>
    %351 = stablehlo.broadcast_in_dim %343, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %352 = stablehlo.subtract %339, %351 : tensor<1x128x512xf32>
    %353 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %354 = stablehlo.add %350, %353 : tensor<1x128x1xf32>
    %355 = stablehlo.rsqrt %354 : tensor<1x128x1xf32>
    %356 = stablehlo.broadcast_in_dim %355, dims = [0, 1, 2] : (tensor<1x128x1xf32>) -> tensor<1x128x512xf32>
    %357 = stablehlo.multiply %352, %356 : tensor<1x128x512xf32>
    %358 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %359 = stablehlo.broadcast_in_dim %358, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %360 = stablehlo.multiply %357, %359 : tensor<1x128x512xf32>
    %361 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<512xf32>) -> tensor<1x1x512xf32>
    %362 = stablehlo.broadcast_in_dim %361, dims = [0, 1, 2] : (tensor<1x1x512xf32>) -> tensor<1x128x512xf32>
    %363 = stablehlo.add %360, %362 : tensor<1x128x512xf32>
    %364 = stablehlo.dot_general %363, %arg37, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf32>, tensor<512x2048xf32>) -> tensor<1x128x2048xf32>
    return %364 : tensor<1x128x2048xf32>
  }
}
