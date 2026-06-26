module @IrToHlo.597 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2048x512xf32>, %arg1: tensor<512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x768xf32>, %arg4: tensor<768xf32>, %arg5: tensor<768x512xf32>, %arg6: tensor<512xf32>, %arg7: tensor<512xf32>, %arg8: tensor<512x512xf32>, %arg9: tensor<1536xf32>, %arg10: tensor<1536x512xf32>, %arg11: tensor<512xf32>, %arg12: tensor<512xf32>, %arg13: tensor<512x768xf32>, %arg14: tensor<768xf32>, %arg15: tensor<768x512xf32>, %arg16: tensor<512xf32>, %arg17: tensor<512xf32>, %arg18: tensor<512x512xf32>, %arg19: tensor<1536xf32>, %arg20: tensor<1536x512xf32>, %arg21: tensor<512xf32>, %arg22: tensor<512xf32>, %arg23: tensor<512x768xf32>, %arg24: tensor<768xf32>, %arg25: tensor<768x512xf32>, %arg26: tensor<512xf32>, %arg27: tensor<512xf32>, %arg28: tensor<512x512xf32>, %arg29: tensor<1536xf32>, %arg30: tensor<1536x512xf32>, %arg31: tensor<512xf32>, %arg32: tensor<1x128xi64>, %arg33: tensor<2048x512xf32>, %arg34: tensor<512xf32>, %arg35: tensor<512xf32>, %arg36: tensor<512xf32>, %arg37: tensor<512xf32>, %arg38: tensor<512xf32>, %arg39: tensor<512xf32>, %arg40: tensor<512xf32>) -> tensor<1x128x2048xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1x128x768xf32>
    %cst_0 = stablehlo.constant dense<0.707106769> : tensor<1x128x768xf32>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<1x128x768xf32>
    %cst_2 = stablehlo.constant dense<0.0883883461> : tensor<1x4x128x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<128xf32>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg32 : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = stablehlo.convert %0 : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "stablehlo.gather"(%arg33, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<2048x512xf32>, tensor<128xui32>) -> tensor<128x512xf32>
    %3 = stablehlo.reshape %2 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%3, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %4 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %5 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %6 = stablehlo.multiply %output, %5 : tensor<1x128x512xf32>
    %7 = stablehlo.add %4, %6 : tensor<1x128x512xf32>
    %8 = stablehlo.reshape %7 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %9 = stablehlo.transpose %arg30, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,1536]{0,1}"} : (tensor<1536x512xf32>) -> tensor<512x1536xf32>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x1536xf32>) -> tensor<128x1536xf32>
    %11 = stablehlo.broadcast_in_dim %arg29, dims = [1] : (tensor<1536xf32>) -> tensor<128x1536xf32>
    %12 = stablehlo.add %10, %11 : tensor<128x1536xf32>
    %13 = stablehlo.reshape %12 : (tensor<128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %14 = stablehlo.transpose %13, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf32>) -> tensor<3x1x4x128x128xf32>
    %15 = stablehlo.slice %14 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %16 = stablehlo.reshape %15 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %17 = stablehlo.slice %14 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %18 = stablehlo.reshape %17 : (tensor<1x1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %19 = stablehlo.transpose %18, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %20 = stablehlo.reshape %19 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %21 = stablehlo.dot_general %16, %20, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %22 = stablehlo.reshape %21 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %23 = stablehlo.multiply %22, %cst_2 : tensor<1x4x128x128xf32>
    %24 = stablehlo.reduce(%23 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %26 = stablehlo.subtract %23, %25 : tensor<1x4x128x128xf32>
    %27 = stablehlo.exponential %26 : tensor<1x4x128x128xf32>
    %28 = stablehlo.reduce(%27 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %30 = stablehlo.divide %27, %29 : tensor<1x4x128x128xf32>
    %31 = stablehlo.reshape %30 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %32 = stablehlo.slice %14 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %33 = stablehlo.reshape %32 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %34 = stablehlo.dot_general %31, %33, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %35 = stablehlo.reshape %34 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %36 = stablehlo.transpose %35, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %37 = stablehlo.reshape %36 : (tensor<1x128x4x128xf32>) -> tensor<128x512xf32>
    %38 = stablehlo.transpose %arg28, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,512]{0,1}"} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %39 = stablehlo.dot_general %37, %38, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %40 = stablehlo.broadcast_in_dim %arg27, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %41 = stablehlo.add %39, %40 : tensor<128x512xf32>
    %42 = stablehlo.reshape %41 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %43 = stablehlo.add %3, %42 : tensor<1x128x512xf32>
    %output_7, %batch_mean_8, %batch_var_9 = "stablehlo.batch_norm_training"(%43, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %44 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %45 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %46 = stablehlo.multiply %output_7, %45 : tensor<1x128x512xf32>
    %47 = stablehlo.add %44, %46 : tensor<1x128x512xf32>
    %48 = stablehlo.reshape %47 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %49 = stablehlo.transpose %arg25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,768]{0,1}"} : (tensor<768x512xf32>) -> tensor<512x768xf32>
    %50 = stablehlo.dot_general %48, %49, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x768xf32>) -> tensor<128x768xf32>
    %51 = stablehlo.broadcast_in_dim %arg24, dims = [1] : (tensor<768xf32>) -> tensor<128x768xf32>
    %52 = stablehlo.add %50, %51 : tensor<128x768xf32>
    %53 = stablehlo.reshape %52 : (tensor<128x768xf32>) -> tensor<1x128x768xf32>
    %54 = stablehlo.multiply %53, %cst_1 : tensor<1x128x768xf32>
    %55 = stablehlo.multiply %53, %cst_0 : tensor<1x128x768xf32>
    %56 = stablehlo.custom_call @mhlo.erf(%55) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %57 = stablehlo.add %56, %cst : tensor<1x128x768xf32>
    %58 = stablehlo.multiply %54, %57 : tensor<1x128x768xf32>
    %59 = stablehlo.reshape %58 : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
    %60 = stablehlo.transpose %arg23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,512]{0,1}"} : (tensor<512x768xf32>) -> tensor<768x512xf32>
    %61 = stablehlo.dot_general %59, %60, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf32>, tensor<768x512xf32>) -> tensor<128x512xf32>
    %62 = stablehlo.broadcast_in_dim %arg22, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %63 = stablehlo.add %61, %62 : tensor<128x512xf32>
    %64 = stablehlo.reshape %63 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %65 = stablehlo.add %43, %64 : tensor<1x128x512xf32>
    %output_10, %batch_mean_11, %batch_var_12 = "stablehlo.batch_norm_training"(%65, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %66 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %67 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %68 = stablehlo.multiply %output_10, %67 : tensor<1x128x512xf32>
    %69 = stablehlo.add %66, %68 : tensor<1x128x512xf32>
    %70 = stablehlo.reshape %69 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %71 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,1536]{0,1}"} : (tensor<1536x512xf32>) -> tensor<512x1536xf32>
    %72 = stablehlo.dot_general %70, %71, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x1536xf32>) -> tensor<128x1536xf32>
    %73 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<1536xf32>) -> tensor<128x1536xf32>
    %74 = stablehlo.add %72, %73 : tensor<128x1536xf32>
    %75 = stablehlo.reshape %74 : (tensor<128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %76 = stablehlo.transpose %75, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf32>) -> tensor<3x1x4x128x128xf32>
    %77 = stablehlo.slice %76 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %78 = stablehlo.reshape %77 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %79 = stablehlo.slice %76 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %80 = stablehlo.reshape %79 : (tensor<1x1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %81 = stablehlo.transpose %80, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %82 = stablehlo.reshape %81 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %83 = stablehlo.dot_general %78, %82, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %84 = stablehlo.reshape %83 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %85 = stablehlo.multiply %84, %cst_2 : tensor<1x4x128x128xf32>
    %86 = stablehlo.reduce(%85 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %87 = stablehlo.broadcast_in_dim %86, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %88 = stablehlo.subtract %85, %87 : tensor<1x4x128x128xf32>
    %89 = stablehlo.exponential %88 : tensor<1x4x128x128xf32>
    %90 = stablehlo.reduce(%89 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %92 = stablehlo.divide %89, %91 : tensor<1x4x128x128xf32>
    %93 = stablehlo.reshape %92 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %94 = stablehlo.slice %76 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %95 = stablehlo.reshape %94 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %96 = stablehlo.dot_general %93, %95, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %97 = stablehlo.reshape %96 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %98 = stablehlo.transpose %97, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %99 = stablehlo.reshape %98 : (tensor<1x128x4x128xf32>) -> tensor<128x512xf32>
    %100 = stablehlo.transpose %arg18, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,512]{0,1}"} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %101 = stablehlo.dot_general %99, %100, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %102 = stablehlo.broadcast_in_dim %arg17, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %103 = stablehlo.add %101, %102 : tensor<128x512xf32>
    %104 = stablehlo.reshape %103 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %105 = stablehlo.add %65, %104 : tensor<1x128x512xf32>
    %output_13, %batch_mean_14, %batch_var_15 = "stablehlo.batch_norm_training"(%105, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %106 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %107 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %108 = stablehlo.multiply %output_13, %107 : tensor<1x128x512xf32>
    %109 = stablehlo.add %106, %108 : tensor<1x128x512xf32>
    %110 = stablehlo.reshape %109 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %111 = stablehlo.transpose %arg15, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,768]{0,1}"} : (tensor<768x512xf32>) -> tensor<512x768xf32>
    %112 = stablehlo.dot_general %110, %111, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x768xf32>) -> tensor<128x768xf32>
    %113 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<768xf32>) -> tensor<128x768xf32>
    %114 = stablehlo.add %112, %113 : tensor<128x768xf32>
    %115 = stablehlo.reshape %114 : (tensor<128x768xf32>) -> tensor<1x128x768xf32>
    %116 = stablehlo.multiply %115, %cst_1 : tensor<1x128x768xf32>
    %117 = stablehlo.multiply %115, %cst_0 : tensor<1x128x768xf32>
    %118 = stablehlo.custom_call @mhlo.erf(%117) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %119 = stablehlo.add %118, %cst : tensor<1x128x768xf32>
    %120 = stablehlo.multiply %116, %119 : tensor<1x128x768xf32>
    %121 = stablehlo.reshape %120 : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
    %122 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,512]{0,1}"} : (tensor<512x768xf32>) -> tensor<768x512xf32>
    %123 = stablehlo.dot_general %121, %122, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf32>, tensor<768x512xf32>) -> tensor<128x512xf32>
    %124 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %125 = stablehlo.add %123, %124 : tensor<128x512xf32>
    %126 = stablehlo.reshape %125 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %127 = stablehlo.add %105, %126 : tensor<1x128x512xf32>
    %output_16, %batch_mean_17, %batch_var_18 = "stablehlo.batch_norm_training"(%127, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %128 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %129 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %130 = stablehlo.multiply %output_16, %129 : tensor<1x128x512xf32>
    %131 = stablehlo.add %128, %130 : tensor<1x128x512xf32>
    %132 = stablehlo.reshape %131 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %133 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,1536]{0,1}"} : (tensor<1536x512xf32>) -> tensor<512x1536xf32>
    %134 = stablehlo.dot_general %132, %133, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x1536xf32>) -> tensor<128x1536xf32>
    %135 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1536xf32>) -> tensor<128x1536xf32>
    %136 = stablehlo.add %134, %135 : tensor<128x1536xf32>
    %137 = stablehlo.reshape %136 : (tensor<128x1536xf32>) -> tensor<1x128x3x4x128xf32>
    %138 = stablehlo.transpose %137, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f32[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf32>) -> tensor<3x1x4x128x128xf32>
    %139 = stablehlo.slice %138 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %140 = stablehlo.reshape %139 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %141 = stablehlo.slice %138 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %142 = stablehlo.reshape %141 : (tensor<1x1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %143 = stablehlo.transpose %142, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f32[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %144 = stablehlo.reshape %143 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %145 = stablehlo.dot_general %140, %144, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %146 = stablehlo.reshape %145 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %147 = stablehlo.multiply %146, %cst_2 : tensor<1x4x128x128xf32>
    %148 = stablehlo.reduce(%147 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %149 = stablehlo.broadcast_in_dim %148, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %150 = stablehlo.subtract %147, %149 : tensor<1x4x128x128xf32>
    %151 = stablehlo.exponential %150 : tensor<1x4x128x128xf32>
    %152 = stablehlo.reduce(%151 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf32>, tensor<f32>) -> tensor<1x4x128xf32>
    %153 = stablehlo.broadcast_in_dim %152, dims = [0, 1, 2] : (tensor<1x4x128xf32>) -> tensor<1x4x128x128xf32>
    %154 = stablehlo.divide %151, %153 : tensor<1x4x128x128xf32>
    %155 = stablehlo.reshape %154 : (tensor<1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %156 = stablehlo.slice %138 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf32>) -> tensor<1x1x4x128x128xf32>
    %157 = stablehlo.reshape %156 : (tensor<1x1x4x128x128xf32>) -> tensor<4x128x128xf32>
    %158 = stablehlo.dot_general %155, %157, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf32>, tensor<4x128x128xf32>) -> tensor<4x128x128xf32>
    %159 = stablehlo.reshape %158 : (tensor<4x128x128xf32>) -> tensor<1x4x128x128xf32>
    %160 = stablehlo.transpose %159, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f32[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf32>) -> tensor<1x128x4x128xf32>
    %161 = stablehlo.reshape %160 : (tensor<1x128x4x128xf32>) -> tensor<128x512xf32>
    %162 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,512]{0,1}"} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %163 = stablehlo.dot_general %161, %162, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x512xf32>) -> tensor<128x512xf32>
    %164 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %165 = stablehlo.add %163, %164 : tensor<128x512xf32>
    %166 = stablehlo.reshape %165 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %167 = stablehlo.add %127, %166 : tensor<1x128x512xf32>
    %output_19, %batch_mean_20, %batch_var_21 = "stablehlo.batch_norm_training"(%167, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %168 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %169 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %170 = stablehlo.multiply %output_19, %169 : tensor<1x128x512xf32>
    %171 = stablehlo.add %168, %170 : tensor<1x128x512xf32>
    %172 = stablehlo.reshape %171 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %173 = stablehlo.transpose %arg5, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,768]{0,1}"} : (tensor<768x512xf32>) -> tensor<512x768xf32>
    %174 = stablehlo.dot_general %172, %173, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x768xf32>) -> tensor<128x768xf32>
    %175 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf32>) -> tensor<128x768xf32>
    %176 = stablehlo.add %174, %175 : tensor<128x768xf32>
    %177 = stablehlo.reshape %176 : (tensor<128x768xf32>) -> tensor<1x128x768xf32>
    %178 = stablehlo.multiply %177, %cst_1 : tensor<1x128x768xf32>
    %179 = stablehlo.multiply %177, %cst_0 : tensor<1x128x768xf32>
    %180 = stablehlo.custom_call @mhlo.erf(%179) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
    %181 = stablehlo.add %180, %cst : tensor<1x128x768xf32>
    %182 = stablehlo.multiply %178, %181 : tensor<1x128x768xf32>
    %183 = stablehlo.reshape %182 : (tensor<1x128x768xf32>) -> tensor<128x768xf32>
    %184 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[768,512]{0,1}"} : (tensor<512x768xf32>) -> tensor<768x512xf32>
    %185 = stablehlo.dot_general %183, %184, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf32>, tensor<768x512xf32>) -> tensor<128x512xf32>
    %186 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<128x512xf32>
    %187 = stablehlo.add %185, %186 : tensor<128x512xf32>
    %188 = stablehlo.reshape %187 : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
    %189 = stablehlo.add %167, %188 : tensor<1x128x512xf32>
    %output_22, %batch_mean_23, %batch_var_24 = "stablehlo.batch_norm_training"(%189, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<1x128x512xf32>, tensor<128xf32>, tensor<128xf32>)
    %190 = stablehlo.broadcast_in_dim %arg40, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %191 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<512xf32>) -> tensor<1x128x512xf32>
    %192 = stablehlo.multiply %output_22, %191 : tensor<1x128x512xf32>
    %193 = stablehlo.add %190, %192 : tensor<1x128x512xf32>
    %194 = stablehlo.reshape %193 : (tensor<1x128x512xf32>) -> tensor<128x512xf32>
    %195 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f32[512,2048]{0,1}"} : (tensor<2048x512xf32>) -> tensor<512x2048xf32>
    %196 = stablehlo.dot_general %194, %195, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf32>, tensor<512x2048xf32>) -> tensor<128x2048xf32>
    %197 = stablehlo.reshape %196 : (tensor<128x2048xf32>) -> tensor<1x128x2048xf32>
    return %197 : tensor<1x128x2048xf32>
  }
}
