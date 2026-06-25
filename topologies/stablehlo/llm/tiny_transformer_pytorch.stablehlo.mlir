module @IrToHlo.603 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2048x512xf16>, %arg1: tensor<512xf16>, %arg2: tensor<512xf16>, %arg3: tensor<512x768xf16>, %arg4: tensor<768xf16>, %arg5: tensor<768x512xf16>, %arg6: tensor<512xf16>, %arg7: tensor<512xf16>, %arg8: tensor<512x512xf16>, %arg9: tensor<1536xf16>, %arg10: tensor<1536x512xf16>, %arg11: tensor<512xf16>, %arg12: tensor<512xf16>, %arg13: tensor<512x768xf16>, %arg14: tensor<768xf16>, %arg15: tensor<768x512xf16>, %arg16: tensor<512xf16>, %arg17: tensor<512xf16>, %arg18: tensor<512x512xf16>, %arg19: tensor<1536xf16>, %arg20: tensor<1536x512xf16>, %arg21: tensor<512xf16>, %arg22: tensor<512xf16>, %arg23: tensor<512x768xf16>, %arg24: tensor<768xf16>, %arg25: tensor<768x512xf16>, %arg26: tensor<512xf16>, %arg27: tensor<512xf16>, %arg28: tensor<512x512xf16>, %arg29: tensor<1536xf16>, %arg30: tensor<1536x512xf16>, %arg31: tensor<512xf16>, %arg32: tensor<1x128xi64>, %arg33: tensor<2048x512xf16>, %arg34: tensor<512xf16>, %arg35: tensor<512xf16>, %arg36: tensor<512xf16>, %arg37: tensor<512xf16>, %arg38: tensor<512xf16>, %arg39: tensor<512xf16>, %arg40: tensor<512xf16>) -> tensor<1x128x2048xf16> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<1x128x768xf16>
    %cst_0 = stablehlo.constant dense<7.070310e-01> : tensor<1x128x768xf16>
    %cst_1 = stablehlo.constant dense<5.000000e-01> : tensor<1x128x768xf16>
    %cst_2 = stablehlo.constant dense<0.0883883461> : tensor<1x4x128x128xf32>
    %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<128xf16>
    %cst_4 = stablehlo.constant dense<1.000000e+00> : tensor<128xf16>
    %cst_5 = stablehlo.constant dense<0xFC00> : tensor<f16>
    %cst_6 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %0 = stablehlo.reshape %arg32 : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = stablehlo.convert %0 : (tensor<128xi64>) -> tensor<128xui32>
    %2 = "stablehlo.gather"(%arg33, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 512>}> : (tensor<2048x512xf16>, tensor<128xui32>) -> tensor<128x512xf16>
    %3 = stablehlo.reshape %2 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%3, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %4 = stablehlo.broadcast_in_dim %arg34, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %5 = stablehlo.broadcast_in_dim %arg31, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %6 = stablehlo.multiply %output, %5 : tensor<1x128x512xf16>
    %7 = stablehlo.add %4, %6 : tensor<1x128x512xf16>
    %8 = stablehlo.reshape %7 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %9 = stablehlo.transpose %arg30, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,1536]{0,1}"} : (tensor<1536x512xf16>) -> tensor<512x1536xf16>
    %10 = stablehlo.dot_general %8, %9, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x1536xf16>) -> tensor<128x1536xf16>
    %11 = stablehlo.broadcast_in_dim %arg29, dims = [1] : (tensor<1536xf16>) -> tensor<128x1536xf16>
    %12 = stablehlo.add %10, %11 : tensor<128x1536xf16>
    %13 = stablehlo.reshape %12 : (tensor<128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %14 = stablehlo.transpose %13, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f16[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf16>) -> tensor<3x1x4x128x128xf16>
    %15 = stablehlo.slice %14 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %16 = stablehlo.reshape %15 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %17 = stablehlo.slice %14 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %18 = stablehlo.reshape %17 : (tensor<1x1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %19 = stablehlo.transpose %18, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f16[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %20 = stablehlo.reshape %19 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %21 = stablehlo.dot_general %16, %20, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %22 = stablehlo.reshape %21 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %23 = stablehlo.convert %22 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %24 = stablehlo.multiply %23, %cst_2 : tensor<1x4x128x128xf32>
    %25 = stablehlo.convert %24 : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf16>
    %26 = stablehlo.reduce(%25 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %28 = stablehlo.subtract %25, %27 : tensor<1x4x128x128xf16>
    %29 = stablehlo.exponential %28 : tensor<1x4x128x128xf16>
    %30 = stablehlo.reduce(%29 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %31 = stablehlo.broadcast_in_dim %30, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %32 = stablehlo.divide %29, %31 : tensor<1x4x128x128xf16>
    %33 = stablehlo.reshape %32 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %34 = stablehlo.slice %14 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %35 = stablehlo.reshape %34 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %36 = stablehlo.dot_general %33, %35, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %37 = stablehlo.reshape %36 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %38 = stablehlo.transpose %37, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f16[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %39 = stablehlo.reshape %38 : (tensor<1x128x4x128xf16>) -> tensor<128x512xf16>
    %40 = stablehlo.transpose %arg28, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,512]{0,1}"} : (tensor<512x512xf16>) -> tensor<512x512xf16>
    %41 = stablehlo.dot_general %39, %40, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x512xf16>) -> tensor<128x512xf16>
    %42 = stablehlo.broadcast_in_dim %arg27, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %43 = stablehlo.add %41, %42 : tensor<128x512xf16>
    %44 = stablehlo.reshape %43 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %45 = stablehlo.add %3, %44 : tensor<1x128x512xf16>
    %output_7, %batch_mean_8, %batch_var_9 = "stablehlo.batch_norm_training"(%45, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %46 = stablehlo.broadcast_in_dim %arg35, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %47 = stablehlo.broadcast_in_dim %arg26, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %48 = stablehlo.multiply %output_7, %47 : tensor<1x128x512xf16>
    %49 = stablehlo.add %46, %48 : tensor<1x128x512xf16>
    %50 = stablehlo.reshape %49 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %51 = stablehlo.transpose %arg25, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,768]{0,1}"} : (tensor<768x512xf16>) -> tensor<512x768xf16>
    %52 = stablehlo.dot_general %50, %51, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x768xf16>) -> tensor<128x768xf16>
    %53 = stablehlo.broadcast_in_dim %arg24, dims = [1] : (tensor<768xf16>) -> tensor<128x768xf16>
    %54 = stablehlo.add %52, %53 : tensor<128x768xf16>
    %55 = stablehlo.reshape %54 : (tensor<128x768xf16>) -> tensor<1x128x768xf16>
    %56 = stablehlo.multiply %55, %cst_1 : tensor<1x128x768xf16>
    %57 = stablehlo.multiply %55, %cst_0 : tensor<1x128x768xf16>
    %58 = stablehlo.custom_call @mhlo.erf(%57) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    %59 = stablehlo.add %58, %cst : tensor<1x128x768xf16>
    %60 = stablehlo.multiply %56, %59 : tensor<1x128x768xf16>
    %61 = stablehlo.reshape %60 : (tensor<1x128x768xf16>) -> tensor<128x768xf16>
    %62 = stablehlo.transpose %arg23, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[768,512]{0,1}"} : (tensor<512x768xf16>) -> tensor<768x512xf16>
    %63 = stablehlo.dot_general %61, %62, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf16>, tensor<768x512xf16>) -> tensor<128x512xf16>
    %64 = stablehlo.broadcast_in_dim %arg22, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %65 = stablehlo.add %63, %64 : tensor<128x512xf16>
    %66 = stablehlo.reshape %65 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %67 = stablehlo.add %45, %66 : tensor<1x128x512xf16>
    %output_10, %batch_mean_11, %batch_var_12 = "stablehlo.batch_norm_training"(%67, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %68 = stablehlo.broadcast_in_dim %arg36, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %69 = stablehlo.broadcast_in_dim %arg21, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %70 = stablehlo.multiply %output_10, %69 : tensor<1x128x512xf16>
    %71 = stablehlo.add %68, %70 : tensor<1x128x512xf16>
    %72 = stablehlo.reshape %71 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %73 = stablehlo.transpose %arg20, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,1536]{0,1}"} : (tensor<1536x512xf16>) -> tensor<512x1536xf16>
    %74 = stablehlo.dot_general %72, %73, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x1536xf16>) -> tensor<128x1536xf16>
    %75 = stablehlo.broadcast_in_dim %arg19, dims = [1] : (tensor<1536xf16>) -> tensor<128x1536xf16>
    %76 = stablehlo.add %74, %75 : tensor<128x1536xf16>
    %77 = stablehlo.reshape %76 : (tensor<128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %78 = stablehlo.transpose %77, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f16[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf16>) -> tensor<3x1x4x128x128xf16>
    %79 = stablehlo.slice %78 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %80 = stablehlo.reshape %79 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %81 = stablehlo.slice %78 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %82 = stablehlo.reshape %81 : (tensor<1x1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %83 = stablehlo.transpose %82, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f16[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %84 = stablehlo.reshape %83 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %85 = stablehlo.dot_general %80, %84, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %86 = stablehlo.reshape %85 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %87 = stablehlo.convert %86 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %88 = stablehlo.multiply %87, %cst_2 : tensor<1x4x128x128xf32>
    %89 = stablehlo.convert %88 : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf16>
    %90 = stablehlo.reduce(%89 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %91 = stablehlo.broadcast_in_dim %90, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %92 = stablehlo.subtract %89, %91 : tensor<1x4x128x128xf16>
    %93 = stablehlo.exponential %92 : tensor<1x4x128x128xf16>
    %94 = stablehlo.reduce(%93 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %95 = stablehlo.broadcast_in_dim %94, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %96 = stablehlo.divide %93, %95 : tensor<1x4x128x128xf16>
    %97 = stablehlo.reshape %96 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %98 = stablehlo.slice %78 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %99 = stablehlo.reshape %98 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %100 = stablehlo.dot_general %97, %99, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %101 = stablehlo.reshape %100 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %102 = stablehlo.transpose %101, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f16[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %103 = stablehlo.reshape %102 : (tensor<1x128x4x128xf16>) -> tensor<128x512xf16>
    %104 = stablehlo.transpose %arg18, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,512]{0,1}"} : (tensor<512x512xf16>) -> tensor<512x512xf16>
    %105 = stablehlo.dot_general %103, %104, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x512xf16>) -> tensor<128x512xf16>
    %106 = stablehlo.broadcast_in_dim %arg17, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %107 = stablehlo.add %105, %106 : tensor<128x512xf16>
    %108 = stablehlo.reshape %107 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %109 = stablehlo.add %67, %108 : tensor<1x128x512xf16>
    %output_13, %batch_mean_14, %batch_var_15 = "stablehlo.batch_norm_training"(%109, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %110 = stablehlo.broadcast_in_dim %arg37, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %111 = stablehlo.broadcast_in_dim %arg16, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %112 = stablehlo.multiply %output_13, %111 : tensor<1x128x512xf16>
    %113 = stablehlo.add %110, %112 : tensor<1x128x512xf16>
    %114 = stablehlo.reshape %113 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %115 = stablehlo.transpose %arg15, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,768]{0,1}"} : (tensor<768x512xf16>) -> tensor<512x768xf16>
    %116 = stablehlo.dot_general %114, %115, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x768xf16>) -> tensor<128x768xf16>
    %117 = stablehlo.broadcast_in_dim %arg14, dims = [1] : (tensor<768xf16>) -> tensor<128x768xf16>
    %118 = stablehlo.add %116, %117 : tensor<128x768xf16>
    %119 = stablehlo.reshape %118 : (tensor<128x768xf16>) -> tensor<1x128x768xf16>
    %120 = stablehlo.multiply %119, %cst_1 : tensor<1x128x768xf16>
    %121 = stablehlo.multiply %119, %cst_0 : tensor<1x128x768xf16>
    %122 = stablehlo.custom_call @mhlo.erf(%121) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    %123 = stablehlo.add %122, %cst : tensor<1x128x768xf16>
    %124 = stablehlo.multiply %120, %123 : tensor<1x128x768xf16>
    %125 = stablehlo.reshape %124 : (tensor<1x128x768xf16>) -> tensor<128x768xf16>
    %126 = stablehlo.transpose %arg13, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[768,512]{0,1}"} : (tensor<512x768xf16>) -> tensor<768x512xf16>
    %127 = stablehlo.dot_general %125, %126, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf16>, tensor<768x512xf16>) -> tensor<128x512xf16>
    %128 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %129 = stablehlo.add %127, %128 : tensor<128x512xf16>
    %130 = stablehlo.reshape %129 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %131 = stablehlo.add %109, %130 : tensor<1x128x512xf16>
    %output_16, %batch_mean_17, %batch_var_18 = "stablehlo.batch_norm_training"(%131, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %132 = stablehlo.broadcast_in_dim %arg38, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %133 = stablehlo.broadcast_in_dim %arg11, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %134 = stablehlo.multiply %output_16, %133 : tensor<1x128x512xf16>
    %135 = stablehlo.add %132, %134 : tensor<1x128x512xf16>
    %136 = stablehlo.reshape %135 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %137 = stablehlo.transpose %arg10, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,1536]{0,1}"} : (tensor<1536x512xf16>) -> tensor<512x1536xf16>
    %138 = stablehlo.dot_general %136, %137, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x1536xf16>) -> tensor<128x1536xf16>
    %139 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<1536xf16>) -> tensor<128x1536xf16>
    %140 = stablehlo.add %138, %139 : tensor<128x1536xf16>
    %141 = stablehlo.reshape %140 : (tensor<128x1536xf16>) -> tensor<1x128x3x4x128xf16>
    %142 = stablehlo.transpose %141, dims = [2, 0, 3, 1, 4] {result_layout = dense<[4, 2, 0, 3, 1]> : tensor<5xindex>, xla_shape = "f16[3,1,4,128,128]{4,2,0,3,1}"} : (tensor<1x128x3x4x128xf16>) -> tensor<3x1x4x128x128xf16>
    %143 = stablehlo.slice %142 [0:1, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %144 = stablehlo.reshape %143 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %145 = stablehlo.slice %142 [1:2, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %146 = stablehlo.reshape %145 : (tensor<1x1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %147 = stablehlo.transpose %146, dims = [0, 1, 3, 2] {result_layout = dense<[2, 3, 1, 0]> : tensor<4xindex>, xla_shape = "f16[1,4,128,128]{2,3,1,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %148 = stablehlo.reshape %147 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %149 = stablehlo.dot_general %144, %148, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %150 = stablehlo.reshape %149 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %151 = stablehlo.convert %150 : (tensor<1x4x128x128xf16>) -> tensor<1x4x128x128xf32>
    %152 = stablehlo.multiply %151, %cst_2 : tensor<1x4x128x128xf32>
    %153 = stablehlo.convert %152 : (tensor<1x4x128x128xf32>) -> tensor<1x4x128x128xf16>
    %154 = stablehlo.reduce(%153 init: %cst_5) applies stablehlo.maximum across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %155 = stablehlo.broadcast_in_dim %154, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %156 = stablehlo.subtract %153, %155 : tensor<1x4x128x128xf16>
    %157 = stablehlo.exponential %156 : tensor<1x4x128x128xf16>
    %158 = stablehlo.reduce(%157 init: %cst_6) applies stablehlo.add across dimensions = [3] : (tensor<1x4x128x128xf16>, tensor<f16>) -> tensor<1x4x128xf16>
    %159 = stablehlo.broadcast_in_dim %158, dims = [0, 1, 2] : (tensor<1x4x128xf16>) -> tensor<1x4x128x128xf16>
    %160 = stablehlo.divide %157, %159 : tensor<1x4x128x128xf16>
    %161 = stablehlo.reshape %160 : (tensor<1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %162 = stablehlo.slice %142 [2:3, 0:1, 0:4, 0:128, 0:128] : (tensor<3x1x4x128x128xf16>) -> tensor<1x1x4x128x128xf16>
    %163 = stablehlo.reshape %162 : (tensor<1x1x4x128x128xf16>) -> tensor<4x128x128xf16>
    %164 = stablehlo.dot_general %161, %163, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<4x128x128xf16>, tensor<4x128x128xf16>) -> tensor<4x128x128xf16>
    %165 = stablehlo.reshape %164 : (tensor<4x128x128xf16>) -> tensor<1x4x128x128xf16>
    %166 = stablehlo.transpose %165, dims = [0, 2, 1, 3] {result_layout = dense<[3, 1, 2, 0]> : tensor<4xindex>, xla_shape = "f16[1,128,4,128]{3,1,2,0}"} : (tensor<1x4x128x128xf16>) -> tensor<1x128x4x128xf16>
    %167 = stablehlo.reshape %166 : (tensor<1x128x4x128xf16>) -> tensor<128x512xf16>
    %168 = stablehlo.transpose %arg8, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,512]{0,1}"} : (tensor<512x512xf16>) -> tensor<512x512xf16>
    %169 = stablehlo.dot_general %167, %168, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x512xf16>) -> tensor<128x512xf16>
    %170 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %171 = stablehlo.add %169, %170 : tensor<128x512xf16>
    %172 = stablehlo.reshape %171 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %173 = stablehlo.add %131, %172 : tensor<1x128x512xf16>
    %output_19, %batch_mean_20, %batch_var_21 = "stablehlo.batch_norm_training"(%173, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %174 = stablehlo.broadcast_in_dim %arg39, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %175 = stablehlo.broadcast_in_dim %arg6, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %176 = stablehlo.multiply %output_19, %175 : tensor<1x128x512xf16>
    %177 = stablehlo.add %174, %176 : tensor<1x128x512xf16>
    %178 = stablehlo.reshape %177 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %179 = stablehlo.transpose %arg5, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,768]{0,1}"} : (tensor<768x512xf16>) -> tensor<512x768xf16>
    %180 = stablehlo.dot_general %178, %179, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x768xf16>) -> tensor<128x768xf16>
    %181 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<768xf16>) -> tensor<128x768xf16>
    %182 = stablehlo.add %180, %181 : tensor<128x768xf16>
    %183 = stablehlo.reshape %182 : (tensor<128x768xf16>) -> tensor<1x128x768xf16>
    %184 = stablehlo.multiply %183, %cst_1 : tensor<1x128x768xf16>
    %185 = stablehlo.multiply %183, %cst_0 : tensor<1x128x768xf16>
    %186 = stablehlo.custom_call @mhlo.erf(%185) {mhlo.attributes = {}, mhlo.version = 1 : i64} : (tensor<1x128x768xf16>) -> tensor<1x128x768xf16>
    %187 = stablehlo.add %186, %cst : tensor<1x128x768xf16>
    %188 = stablehlo.multiply %184, %187 : tensor<1x128x768xf16>
    %189 = stablehlo.reshape %188 : (tensor<1x128x768xf16>) -> tensor<128x768xf16>
    %190 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[768,512]{0,1}"} : (tensor<512x768xf16>) -> tensor<768x512xf16>
    %191 = stablehlo.dot_general %189, %190, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x768xf16>, tensor<768x512xf16>) -> tensor<128x512xf16>
    %192 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf16>) -> tensor<128x512xf16>
    %193 = stablehlo.add %191, %192 : tensor<128x512xf16>
    %194 = stablehlo.reshape %193 : (tensor<128x512xf16>) -> tensor<1x128x512xf16>
    %195 = stablehlo.add %173, %194 : tensor<1x128x512xf16>
    %output_22, %batch_mean_23, %batch_var_24 = "stablehlo.batch_norm_training"(%195, %cst_4, %cst_3) <{epsilon = 9.99999974E-6 : f32, feature_index = 1 : i64}> : (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>) -> (tensor<1x128x512xf16>, tensor<128xf16>, tensor<128xf16>)
    %196 = stablehlo.broadcast_in_dim %arg40, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %197 = stablehlo.broadcast_in_dim %arg1, dims = [2] : (tensor<512xf16>) -> tensor<1x128x512xf16>
    %198 = stablehlo.multiply %output_22, %197 : tensor<1x128x512xf16>
    %199 = stablehlo.add %196, %198 : tensor<1x128x512xf16>
    %200 = stablehlo.reshape %199 : (tensor<1x128x512xf16>) -> tensor<128x512xf16>
    %201 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "f16[512,2048]{0,1}"} : (tensor<2048x512xf16>) -> tensor<512x2048xf16>
    %202 = stablehlo.dot_general %200, %201, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<128x512xf16>, tensor<512x2048xf16>) -> tensor<128x2048xf16>
    %203 = stablehlo.reshape %202 : (tensor<128x2048xf16>) -> tensor<1x128x2048xf16>
    return %203 : tensor<1x128x2048xf16>
  }
}
