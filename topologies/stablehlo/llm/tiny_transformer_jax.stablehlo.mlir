module @jit_forward attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<768xf16>, %arg1: tensor<512x768xf16>, %arg2: tensor<512xf16>, %arg3: tensor<768x512xf16>, %arg4: tensor<512xf16>, %arg5: tensor<512xf16>, %arg6: tensor<512xf16>, %arg7: tensor<512xf16>, %arg8: tensor<512xf16>, %arg9: tensor<512x512xf16>, %arg10: tensor<1536xf16>, %arg11: tensor<512x1536xf16>, %arg12: tensor<768xf16>, %arg13: tensor<512x768xf16>, %arg14: tensor<512xf16>, %arg15: tensor<768x512xf16>, %arg16: tensor<512xf16>, %arg17: tensor<512xf16>, %arg18: tensor<512xf16>, %arg19: tensor<512xf16>, %arg20: tensor<512xf16>, %arg21: tensor<512x512xf16>, %arg22: tensor<1536xf16>, %arg23: tensor<512x1536xf16>, %arg24: tensor<768xf16>, %arg25: tensor<512x768xf16>, %arg26: tensor<512xf16>, %arg27: tensor<768x512xf16>, %arg28: tensor<512xf16>, %arg29: tensor<512xf16>, %arg30: tensor<512xf16>, %arg31: tensor<512xf16>, %arg32: tensor<512xf16>, %arg33: tensor<512x512xf16>, %arg34: tensor<1536xf16>, %arg35: tensor<512x1536xf16>, %arg36: tensor<2048x512xf16>, %arg37: tensor<512x2048xf16>, %arg38: tensor<512xf16>, %arg39: tensor<512xf16>, %arg40: tensor<1x128xi32>) -> (tensor<1x128x2048xf16> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<1x128xi32>
    %1 = stablehlo.compare  LT, %arg40, %0,  SIGNED : (tensor<1x128xi32>, tensor<1x128xi32>) -> tensor<1x128xi1>
    %c_0 = stablehlo.constant dense<2048> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<1x128xi32>
    %3 = stablehlo.add %arg40, %2 : tensor<1x128xi32>
    %4 = stablehlo.select %1, %3, %arg40 : tensor<1x128xi1>, tensor<1x128xi32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x128xi32>) -> tensor<1x128x1xi32>
    %6 = "stablehlo.gather"(%arg36, %5) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<2048x512xf16>, tensor<1x128x1xi32>) -> tensor<1x128x512xf16>
    %7 = stablehlo.convert %6 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%7 init: %cst) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %cst_1 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %11 = stablehlo.divide %9, %10 : tensor<1x128x1xf32>
    %12 = stablehlo.convert %11 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %14 = stablehlo.subtract %6, %13 : tensor<1x128x512xf16>
    %15 = stablehlo.multiply %14, %14 : tensor<1x128x512xf16>
    %16 = stablehlo.convert %15 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.reduce(%16 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %19 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %20 = stablehlo.divide %18, %19 : tensor<1x128x1xf32>
    %21 = stablehlo.convert %20 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %22 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %23 = stablehlo.subtract %6, %22 : tensor<1x128x512xf16>
    %cst_3 = stablehlo.constant dense<1.001360e-05> : tensor<f16>
    %24 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %25 = stablehlo.add %21, %24 : tensor<1x128x1xf16>
    %26 = stablehlo.rsqrt %25 : tensor<1x128x1xf16>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %28 = stablehlo.multiply %23, %27 : tensor<1x128x512xf16>
    %29 = stablehlo.broadcast_in_dim %arg5, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %30 = stablehlo.broadcast_in_dim %29, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %31 = stablehlo.multiply %28, %30 : tensor<1x128x512xf16>
    %32 = stablehlo.broadcast_in_dim %arg4, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %34 = stablehlo.add %31, %33 : tensor<1x128x512xf16>
    %35 = stablehlo.dot_general %34, %arg11, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x1536xf16>) -> tensor<1x128x1536xf16>
    %36 = stablehlo.broadcast_in_dim %arg10, dims = [2] : (tensor<1536xf16>) -> tensor<1x1x1536xf16>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1, 2] : (tensor<1x1x1536xf16>) -> tensor<1x128x1536xf16>
    %38 = stablehlo.add %35, %37 : tensor<1x128x1536xf16>
    %39 = stablehlo.reshape %38 : (tensor<1x128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %40 = stablehlo.slice %39 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %41 = stablehlo.reshape %40 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %42 = stablehlo.transpose %41, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %43 = stablehlo.slice %39 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %44 = stablehlo.reshape %43 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %45 = stablehlo.transpose %44, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %46 = stablehlo.slice %39 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %47 = stablehlo.reshape %46 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %48 = stablehlo.transpose %47, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %49 = stablehlo.transpose %45, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %50 = stablehlo.reshape %42 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %51 = stablehlo.dot_general %50, %49, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %52 = stablehlo.transpose %51, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %cst_4 = stablehlo.constant dense<8.837890e-02> : tensor<f16>
    %53 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f16>) -> tensor<1x4x128x128xf16>
    %54 = stablehlo.multiply %52, %53 : tensor<1x4x128x128xf16>
    %cst_5 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %55 = stablehlo.reduce(%54 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %cst_6 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %56 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f16>) -> tensor<1x4x128xf16>
    %57 = stablehlo.maximum %56, %55 : tensor<1x4x128xf16>
    %58 = stablehlo.broadcast_in_dim %57, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x1xf16>
    %59 = stablehlo.broadcast_in_dim %58, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %60 = stablehlo.subtract %54, %59 : tensor<1x4x128x128xf16>
    %61 = stablehlo.exponential %60 : tensor<1x4x128x128xf16>
    %62 = stablehlo.convert %61 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %63 = stablehlo.reduce(%62 init: %cst_7) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %64 = stablehlo.broadcast_in_dim %63, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %65 = stablehlo.convert %64 : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf16>
    %66 = stablehlo.broadcast_in_dim %65, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %67 = stablehlo.divide %61, %66 : tensor<1x4x128x128xf16>
    %68 = stablehlo.reshape %67 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %69 = stablehlo.dot_general %68, %48, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %70 = stablehlo.transpose %69, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %71 = stablehlo.transpose %70, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %72 = stablehlo.reshape %71 : (tensor<1x128x4x128xf16>) -> tensor<1x128x512xf16>
    %73 = stablehlo.dot_general %72, %arg9, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x512xf16>) -> tensor<1x128x512xf16>
    %74 = stablehlo.broadcast_in_dim %arg8, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %76 = stablehlo.add %73, %75 : tensor<1x128x512xf16>
    %77 = stablehlo.add %6, %76 : tensor<1x128x512xf16>
    %78 = stablehlo.convert %77 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %79 = stablehlo.reduce(%78 init: %cst_8) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %80 = stablehlo.broadcast_in_dim %79, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %81 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %82 = stablehlo.divide %80, %81 : tensor<1x128x1xf32>
    %83 = stablehlo.convert %82 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %84 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %85 = stablehlo.subtract %77, %84 : tensor<1x128x512xf16>
    %86 = stablehlo.multiply %85, %85 : tensor<1x128x512xf16>
    %87 = stablehlo.convert %86 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_9 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %88 = stablehlo.reduce(%87 init: %cst_9) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %89 = stablehlo.broadcast_in_dim %88, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %90 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %91 = stablehlo.divide %89, %90 : tensor<1x128x1xf32>
    %92 = stablehlo.convert %91 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %93 = stablehlo.broadcast_in_dim %83, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %94 = stablehlo.subtract %77, %93 : tensor<1x128x512xf16>
    %95 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %96 = stablehlo.add %92, %95 : tensor<1x128x1xf16>
    %97 = stablehlo.rsqrt %96 : tensor<1x128x1xf16>
    %98 = stablehlo.broadcast_in_dim %97, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %99 = stablehlo.multiply %94, %98 : tensor<1x128x512xf16>
    %100 = stablehlo.broadcast_in_dim %arg7, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %101 = stablehlo.broadcast_in_dim %100, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %102 = stablehlo.multiply %99, %101 : tensor<1x128x512xf16>
    %103 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %104 = stablehlo.broadcast_in_dim %103, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %105 = stablehlo.add %102, %104 : tensor<1x128x512xf16>
    %106 = stablehlo.dot_general %105, %arg1, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x768xf16>) -> tensor<1x128x768xf16>
    %107 = stablehlo.broadcast_in_dim %arg0, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %108 = stablehlo.broadcast_in_dim %107, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<1x128x768xf16>
    %109 = stablehlo.add %106, %108 : tensor<1x128x768xf16>
    %110 = stablehlo.multiply %109, %109 : tensor<1x128x768xf16>
    %111 = stablehlo.multiply %110, %109 : tensor<1x128x768xf16>
    %cst_10 = stablehlo.constant dense<4.470830e-02> : tensor<f16>
    %112 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %113 = stablehlo.multiply %112, %111 : tensor<1x128x768xf16>
    %114 = stablehlo.add %109, %113 : tensor<1x128x768xf16>
    %cst_11 = stablehlo.constant dense<7.978520e-01> : tensor<f16>
    %115 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %116 = stablehlo.multiply %115, %114 : tensor<1x128x768xf16>
    %117 = stablehlo.tanh %116 : tensor<1x128x768xf16>
    %cst_12 = stablehlo.constant dense<1.000000e+00> : tensor<f16>
    %118 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %119 = stablehlo.add %118, %117 : tensor<1x128x768xf16>
    %cst_13 = stablehlo.constant dense<5.000000e-01> : tensor<f16>
    %120 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %121 = stablehlo.multiply %120, %119 : tensor<1x128x768xf16>
    %122 = stablehlo.multiply %109, %121 : tensor<1x128x768xf16>
    %123 = stablehlo.dot_general %122, %arg3, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf16>, tensor<768x512xf16>) -> tensor<1x128x512xf16>
    %124 = stablehlo.broadcast_in_dim %arg2, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %125 = stablehlo.broadcast_in_dim %124, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %126 = stablehlo.add %123, %125 : tensor<1x128x512xf16>
    %127 = stablehlo.add %77, %126 : tensor<1x128x512xf16>
    %128 = stablehlo.convert %127 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %129 = stablehlo.reduce(%128 init: %cst_14) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %130 = stablehlo.broadcast_in_dim %129, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %131 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %132 = stablehlo.divide %130, %131 : tensor<1x128x1xf32>
    %133 = stablehlo.convert %132 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %134 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %135 = stablehlo.subtract %127, %134 : tensor<1x128x512xf16>
    %136 = stablehlo.multiply %135, %135 : tensor<1x128x512xf16>
    %137 = stablehlo.convert %136 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %138 = stablehlo.reduce(%137 init: %cst_15) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %139 = stablehlo.broadcast_in_dim %138, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %140 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %141 = stablehlo.divide %139, %140 : tensor<1x128x1xf32>
    %142 = stablehlo.convert %141 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %143 = stablehlo.broadcast_in_dim %133, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %144 = stablehlo.subtract %127, %143 : tensor<1x128x512xf16>
    %145 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %146 = stablehlo.add %142, %145 : tensor<1x128x1xf16>
    %147 = stablehlo.rsqrt %146 : tensor<1x128x1xf16>
    %148 = stablehlo.broadcast_in_dim %147, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %149 = stablehlo.multiply %144, %148 : tensor<1x128x512xf16>
    %150 = stablehlo.broadcast_in_dim %arg17, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %151 = stablehlo.broadcast_in_dim %150, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %152 = stablehlo.multiply %149, %151 : tensor<1x128x512xf16>
    %153 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %154 = stablehlo.broadcast_in_dim %153, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %155 = stablehlo.add %152, %154 : tensor<1x128x512xf16>
    %156 = stablehlo.dot_general %155, %arg23, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x1536xf16>) -> tensor<1x128x1536xf16>
    %157 = stablehlo.broadcast_in_dim %arg22, dims = [2] : (tensor<1536xf16>) -> tensor<1x1x1536xf16>
    %158 = stablehlo.broadcast_in_dim %157, dims = [0, 1, 2] : (tensor<1x1x1536xf16>) -> tensor<1x128x1536xf16>
    %159 = stablehlo.add %156, %158 : tensor<1x128x1536xf16>
    %160 = stablehlo.reshape %159 : (tensor<1x128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %161 = stablehlo.slice %160 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %162 = stablehlo.reshape %161 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %163 = stablehlo.transpose %162, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %164 = stablehlo.slice %160 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %165 = stablehlo.reshape %164 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %166 = stablehlo.transpose %165, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %167 = stablehlo.slice %160 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %168 = stablehlo.reshape %167 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %169 = stablehlo.transpose %168, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %170 = stablehlo.transpose %166, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %171 = stablehlo.reshape %163 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %172 = stablehlo.dot_general %171, %170, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %173 = stablehlo.transpose %172, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %174 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f16>) -> tensor<1x4x128x128xf16>
    %175 = stablehlo.multiply %173, %174 : tensor<1x4x128x128xf16>
    %cst_16 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %176 = stablehlo.reduce(%175 init: %cst_16) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %177 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f16>) -> tensor<1x4x128xf16>
    %178 = stablehlo.maximum %177, %176 : tensor<1x4x128xf16>
    %179 = stablehlo.broadcast_in_dim %178, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x1xf16>
    %180 = stablehlo.broadcast_in_dim %179, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %181 = stablehlo.subtract %175, %180 : tensor<1x4x128x128xf16>
    %182 = stablehlo.exponential %181 : tensor<1x4x128x128xf16>
    %183 = stablehlo.convert %182 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %cst_17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %184 = stablehlo.reduce(%183 init: %cst_17) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %185 = stablehlo.broadcast_in_dim %184, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %186 = stablehlo.convert %185 : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf16>
    %187 = stablehlo.broadcast_in_dim %186, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %188 = stablehlo.divide %182, %187 : tensor<1x4x128x128xf16>
    %189 = stablehlo.reshape %188 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %190 = stablehlo.dot_general %189, %169, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %191 = stablehlo.transpose %190, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %192 = stablehlo.transpose %191, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %193 = stablehlo.reshape %192 : (tensor<1x128x4x128xf16>) -> tensor<1x128x512xf16>
    %194 = stablehlo.dot_general %193, %arg21, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x512xf16>) -> tensor<1x128x512xf16>
    %195 = stablehlo.broadcast_in_dim %arg20, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %196 = stablehlo.broadcast_in_dim %195, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %197 = stablehlo.add %194, %196 : tensor<1x128x512xf16>
    %198 = stablehlo.add %127, %197 : tensor<1x128x512xf16>
    %199 = stablehlo.convert %198 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_18 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %200 = stablehlo.reduce(%199 init: %cst_18) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %201 = stablehlo.broadcast_in_dim %200, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %202 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %203 = stablehlo.divide %201, %202 : tensor<1x128x1xf32>
    %204 = stablehlo.convert %203 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %205 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %206 = stablehlo.subtract %198, %205 : tensor<1x128x512xf16>
    %207 = stablehlo.multiply %206, %206 : tensor<1x128x512xf16>
    %208 = stablehlo.convert %207 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %209 = stablehlo.reduce(%208 init: %cst_19) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %210 = stablehlo.broadcast_in_dim %209, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %211 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %212 = stablehlo.divide %210, %211 : tensor<1x128x1xf32>
    %213 = stablehlo.convert %212 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %214 = stablehlo.broadcast_in_dim %204, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %215 = stablehlo.subtract %198, %214 : tensor<1x128x512xf16>
    %216 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %217 = stablehlo.add %213, %216 : tensor<1x128x1xf16>
    %218 = stablehlo.rsqrt %217 : tensor<1x128x1xf16>
    %219 = stablehlo.broadcast_in_dim %218, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %220 = stablehlo.multiply %215, %219 : tensor<1x128x512xf16>
    %221 = stablehlo.broadcast_in_dim %arg19, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %222 = stablehlo.broadcast_in_dim %221, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %223 = stablehlo.multiply %220, %222 : tensor<1x128x512xf16>
    %224 = stablehlo.broadcast_in_dim %arg18, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %225 = stablehlo.broadcast_in_dim %224, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %226 = stablehlo.add %223, %225 : tensor<1x128x512xf16>
    %227 = stablehlo.dot_general %226, %arg13, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x768xf16>) -> tensor<1x128x768xf16>
    %228 = stablehlo.broadcast_in_dim %arg12, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %229 = stablehlo.broadcast_in_dim %228, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<1x128x768xf16>
    %230 = stablehlo.add %227, %229 : tensor<1x128x768xf16>
    %231 = stablehlo.multiply %230, %230 : tensor<1x128x768xf16>
    %232 = stablehlo.multiply %231, %230 : tensor<1x128x768xf16>
    %233 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %234 = stablehlo.multiply %233, %232 : tensor<1x128x768xf16>
    %235 = stablehlo.add %230, %234 : tensor<1x128x768xf16>
    %236 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %237 = stablehlo.multiply %236, %235 : tensor<1x128x768xf16>
    %238 = stablehlo.tanh %237 : tensor<1x128x768xf16>
    %239 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %240 = stablehlo.add %239, %238 : tensor<1x128x768xf16>
    %241 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %242 = stablehlo.multiply %241, %240 : tensor<1x128x768xf16>
    %243 = stablehlo.multiply %230, %242 : tensor<1x128x768xf16>
    %244 = stablehlo.dot_general %243, %arg15, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf16>, tensor<768x512xf16>) -> tensor<1x128x512xf16>
    %245 = stablehlo.broadcast_in_dim %arg14, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %246 = stablehlo.broadcast_in_dim %245, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %247 = stablehlo.add %244, %246 : tensor<1x128x512xf16>
    %248 = stablehlo.add %198, %247 : tensor<1x128x512xf16>
    %249 = stablehlo.convert %248 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_20 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %250 = stablehlo.reduce(%249 init: %cst_20) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %251 = stablehlo.broadcast_in_dim %250, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %252 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %253 = stablehlo.divide %251, %252 : tensor<1x128x1xf32>
    %254 = stablehlo.convert %253 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %255 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %256 = stablehlo.subtract %248, %255 : tensor<1x128x512xf16>
    %257 = stablehlo.multiply %256, %256 : tensor<1x128x512xf16>
    %258 = stablehlo.convert %257 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %259 = stablehlo.reduce(%258 init: %cst_21) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %260 = stablehlo.broadcast_in_dim %259, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %261 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %262 = stablehlo.divide %260, %261 : tensor<1x128x1xf32>
    %263 = stablehlo.convert %262 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %264 = stablehlo.broadcast_in_dim %254, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %265 = stablehlo.subtract %248, %264 : tensor<1x128x512xf16>
    %266 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %267 = stablehlo.add %263, %266 : tensor<1x128x1xf16>
    %268 = stablehlo.rsqrt %267 : tensor<1x128x1xf16>
    %269 = stablehlo.broadcast_in_dim %268, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %270 = stablehlo.multiply %265, %269 : tensor<1x128x512xf16>
    %271 = stablehlo.broadcast_in_dim %arg29, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %272 = stablehlo.broadcast_in_dim %271, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %273 = stablehlo.multiply %270, %272 : tensor<1x128x512xf16>
    %274 = stablehlo.broadcast_in_dim %arg28, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %275 = stablehlo.broadcast_in_dim %274, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %276 = stablehlo.add %273, %275 : tensor<1x128x512xf16>
    %277 = stablehlo.dot_general %276, %arg35, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x1536xf16>) -> tensor<1x128x1536xf16>
    %278 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<1536xf16>) -> tensor<1x1x1536xf16>
    %279 = stablehlo.broadcast_in_dim %278, dims = [0, 1, 2] : (tensor<1x1x1536xf16>) -> tensor<1x128x1536xf16>
    %280 = stablehlo.add %277, %279 : tensor<1x128x1536xf16>
    %281 = stablehlo.reshape %280 : (tensor<1x128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %282 = stablehlo.slice %281 [0:1, 0:128, 0:1, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %283 = stablehlo.reshape %282 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %284 = stablehlo.transpose %283, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %285 = stablehlo.slice %281 [0:1, 0:128, 1:2, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %286 = stablehlo.reshape %285 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %287 = stablehlo.transpose %286, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %288 = stablehlo.slice %281 [0:1, 0:128, 2:3, 0:4, 0:128] : (tensor<1x128x3x4x128xf16>) -> tensor<1x128x1x4x128xf16>
    %289 = stablehlo.reshape %288 : (tensor<1x128x1x4x128xf16>) -> tensor<1x128x4x128xf16>
    %290 = stablehlo.transpose %289, dims = [0, 2, 1, 3] : (tensor<1x128x4x128xf16>) -> tensor<1x4x128x128xf16>
    %291 = stablehlo.transpose %287, dims = [0, 1, 3, 2] : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %292 = stablehlo.reshape %284 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %293 = stablehlo.dot_general %292, %291, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %294 = stablehlo.transpose %293, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %295 = stablehlo.broadcast_in_dim %cst_4, dims = [] : (tensor<f16>) -> tensor<1x4x128x128xf16>
    %296 = stablehlo.multiply %294, %295 : tensor<1x4x128x128xf16>
    %cst_22 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %297 = stablehlo.reduce(%296 init: %cst_22) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %298 = stablehlo.broadcast_in_dim %cst_6, dims = [] : (tensor<f16>) -> tensor<1x4x128xf16>
    %299 = stablehlo.maximum %298, %297 : tensor<1x4x128xf16>
    %300 = stablehlo.broadcast_in_dim %299, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x1xf16>
    %301 = stablehlo.broadcast_in_dim %300, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %302 = stablehlo.subtract %296, %301 : tensor<1x4x128x128xf16>
    %303 = stablehlo.exponential %302 : tensor<1x4x128x128xf16>
    %304 = stablehlo.convert %303 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %cst_23 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %305 = stablehlo.reduce(%304 init: %cst_23) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %306 = stablehlo.broadcast_in_dim %305, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x1xf32>
    %307 = stablehlo.convert %306 : (tensor<1x4x128x1xf32>) -> tensor<1x4x128x1xf16>
    %308 = stablehlo.broadcast_in_dim %307, dims = [0, 1, 2, 3] : (tensor<1x4x128x1xf16>) -> tensor<1x4x128x128xf16>
    %309 = stablehlo.divide %303, %308 : tensor<1x4x128x128xf16>
    %310 = stablehlo.reshape %309 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %311 = stablehlo.dot_general %310, %290, batching_dims = [0] x [1], contracting_dims = [2] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<1x4x128x128xf16>) -> tensor<4x128x1x128xf16>
    %312 = stablehlo.transpose %311, dims = [2, 0, 1, 3] : (tensor<4x128x1x128xf16>) -> tensor<1x4x128x128xf16>
    %313 = stablehlo.transpose %312, dims = [0, 2, 1, 3] : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %314 = stablehlo.reshape %313 : (tensor<1x128x4x128xf16>) -> tensor<1x128x512xf16>
    %315 = stablehlo.dot_general %314, %arg33, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x512xf16>) -> tensor<1x128x512xf16>
    %316 = stablehlo.broadcast_in_dim %arg32, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %317 = stablehlo.broadcast_in_dim %316, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %318 = stablehlo.add %315, %317 : tensor<1x128x512xf16>
    %319 = stablehlo.add %248, %318 : tensor<1x128x512xf16>
    %320 = stablehlo.convert %319 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %321 = stablehlo.reduce(%320 init: %cst_24) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %322 = stablehlo.broadcast_in_dim %321, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %323 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %324 = stablehlo.divide %322, %323 : tensor<1x128x1xf32>
    %325 = stablehlo.convert %324 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %326 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %327 = stablehlo.subtract %319, %326 : tensor<1x128x512xf16>
    %328 = stablehlo.multiply %327, %327 : tensor<1x128x512xf16>
    %329 = stablehlo.convert %328 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %330 = stablehlo.reduce(%329 init: %cst_25) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %331 = stablehlo.broadcast_in_dim %330, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %332 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %333 = stablehlo.divide %331, %332 : tensor<1x128x1xf32>
    %334 = stablehlo.convert %333 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %335 = stablehlo.broadcast_in_dim %325, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %336 = stablehlo.subtract %319, %335 : tensor<1x128x512xf16>
    %337 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %338 = stablehlo.add %334, %337 : tensor<1x128x1xf16>
    %339 = stablehlo.rsqrt %338 : tensor<1x128x1xf16>
    %340 = stablehlo.broadcast_in_dim %339, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %341 = stablehlo.multiply %336, %340 : tensor<1x128x512xf16>
    %342 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %343 = stablehlo.broadcast_in_dim %342, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %344 = stablehlo.multiply %341, %343 : tensor<1x128x512xf16>
    %345 = stablehlo.broadcast_in_dim %arg30, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %346 = stablehlo.broadcast_in_dim %345, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %347 = stablehlo.add %344, %346 : tensor<1x128x512xf16>
    %348 = stablehlo.dot_general %347, %arg25, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x768xf16>) -> tensor<1x128x768xf16>
    %349 = stablehlo.broadcast_in_dim %arg24, dims = [2] : (tensor<768xf16>) -> tensor<1x1x768xf16>
    %350 = stablehlo.broadcast_in_dim %349, dims = [0, 1, 2] : (tensor<1x1x768xf16>) -> tensor<1x128x768xf16>
    %351 = stablehlo.add %348, %350 : tensor<1x128x768xf16>
    %352 = stablehlo.multiply %351, %351 : tensor<1x128x768xf16>
    %353 = stablehlo.multiply %352, %351 : tensor<1x128x768xf16>
    %354 = stablehlo.broadcast_in_dim %cst_10, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %355 = stablehlo.multiply %354, %353 : tensor<1x128x768xf16>
    %356 = stablehlo.add %351, %355 : tensor<1x128x768xf16>
    %357 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %358 = stablehlo.multiply %357, %356 : tensor<1x128x768xf16>
    %359 = stablehlo.tanh %358 : tensor<1x128x768xf16>
    %360 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %361 = stablehlo.add %360, %359 : tensor<1x128x768xf16>
    %362 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f16>) -> tensor<1x128x768xf16>
    %363 = stablehlo.multiply %362, %361 : tensor<1x128x768xf16>
    %364 = stablehlo.multiply %351, %363 : tensor<1x128x768xf16>
    %365 = stablehlo.dot_general %364, %arg27, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x768xf16>, tensor<768x512xf16>) -> tensor<1x128x512xf16>
    %366 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %367 = stablehlo.broadcast_in_dim %366, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %368 = stablehlo.add %365, %367 : tensor<1x128x512xf16>
    %369 = stablehlo.add %319, %368 : tensor<1x128x512xf16>
    %370 = stablehlo.convert %369 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %371 = stablehlo.reduce(%370 init: %cst_26) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %372 = stablehlo.broadcast_in_dim %371, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %373 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %374 = stablehlo.divide %372, %373 : tensor<1x128x1xf32>
    %375 = stablehlo.convert %374 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %376 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %377 = stablehlo.subtract %369, %376 : tensor<1x128x512xf16>
    %378 = stablehlo.multiply %377, %377 : tensor<1x128x512xf16>
    %379 = stablehlo.convert %378 : (tensor<1x128x512xf16>) -> tensor<1x128x512xf32>
    %cst_27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %380 = stablehlo.reduce(%379 init: %cst_27) applies stablehlo.add across dimensions = [2] : (tensor<1x128x512xf32>, tensor<f32>) -> tensor<1x128xf32>
    %381 = stablehlo.broadcast_in_dim %380, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128x1xf32>
    %382 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<1x128x1xf32>
    %383 = stablehlo.divide %381, %382 : tensor<1x128x1xf32>
    %384 = stablehlo.convert %383 : (tensor<1x128x1xf32>) -> tensor<1x128x1xf16>
    %385 = stablehlo.broadcast_in_dim %375, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %386 = stablehlo.subtract %369, %385 : tensor<1x128x512xf16>
    %387 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f16>) -> tensor<1x128x1xf16>
    %388 = stablehlo.add %384, %387 : tensor<1x128x1xf16>
    %389 = stablehlo.rsqrt %388 : tensor<1x128x1xf16>
    %390 = stablehlo.broadcast_in_dim %389, dims = [0, 1, 2] : (tensor<1x128x1xf16>) -> tensor<1x128x512xf16>
    %391 = stablehlo.multiply %386, %390 : tensor<1x128x512xf16>
    %392 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %393 = stablehlo.broadcast_in_dim %392, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %394 = stablehlo.multiply %391, %393 : tensor<1x128x512xf16>
    %395 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<512xf16>) -> tensor<1x1x512xf16>
    %396 = stablehlo.broadcast_in_dim %395, dims = [0, 1, 2] : (tensor<1x1x512xf16>) -> tensor<1x128x512xf16>
    %397 = stablehlo.add %394, %396 : tensor<1x128x512xf16>
    %398 = stablehlo.dot_general %397, %arg37, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x128x512xf16>, tensor<512x2048xf16>) -> tensor<1x128x2048xf16>
    return %398 : tensor<1x128x2048xf16>
  }
}
