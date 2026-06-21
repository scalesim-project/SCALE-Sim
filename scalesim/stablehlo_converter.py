"""
StableHLO to SCALE-Sim Converter

This module converts StableHLO MLIR operations into SCALE-Sim topology format.
It parses StableHLO operations and translates them into either convolution or GEMM formats
that SCALE-Sim can simulate.
"""

from __future__ import annotations

import os
import sys
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
import pandas as pd

# Try to import the StableHLO parser (now local to scalesim)
try:
    from scalesim.stablehlo_parser import StableHLOParser, OpInfo
    STABLEHLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: StableHLO parser not available: {e}")
    print("Note: JAX/JAXlib are required for MLIR parsing. Install with: pip install jax jaxlib")
    STABLEHLO_AVAILABLE = False
    StableHLOParser = None
    OpInfo = None


class NonComputeOpsStore:
    """
    Storage class for non-convolution and non-GEMM StableHLO operations.
    
    This class keeps track of all operations that are not compute-intensive
    (i.e., not convolutions or matrix multiplications), such as:
    - Element-wise operations (add, subtract, multiply, divide, etc.)
    - Reshape, transpose, broadcast operations
    - Activation functions (relu, sigmoid, tanh, etc.)
    - Reduction operations (reduce_sum, reduce_max, etc.)
    - Other utility operations
    
    Each operation is stored as an OpInfo object preserving all original information.
    """
    
    # Operation categories for classification
    ELEMENTWISE_OPS = {
        'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply', 'stablehlo.divide',
        'stablehlo.maximum', 'stablehlo.minimum', 'stablehlo.power', 'stablehlo.remainder',
        'stablehlo.negate', 'stablehlo.abs', 'stablehlo.sign', 'stablehlo.ceil', 'stablehlo.floor',
        'stablehlo.round_nearest_afz', 'stablehlo.sqrt', 'stablehlo.rsqrt',
        'stablehlo.exp', 'stablehlo.expm1', 'stablehlo.log', 'stablehlo.log_plus_one',
        'stablehlo.sine', 'stablehlo.cosine', 'stablehlo.tanh',
        'stablehlo.compare', 'stablehlo.select', 'stablehlo.clamp',
        'stablehlo.and', 'stablehlo.or', 'stablehlo.xor', 'stablehlo.not',
    }
    
    ACTIVATION_OPS = {
        'stablehlo.logistic',  # sigmoid
        'stablehlo.tanh',
    }
    
    RESHAPE_OPS = {
        'stablehlo.reshape', 'stablehlo.transpose', 'stablehlo.broadcast_in_dim',
        'stablehlo.concatenate', 'stablehlo.slice', 'stablehlo.dynamic_slice',
        'stablehlo.gather', 'stablehlo.scatter', 'stablehlo.pad',
        'stablehlo.reverse', 'stablehlo.bitcast_convert',
    }
    
    REDUCTION_OPS = {
        'stablehlo.reduce', 'stablehlo.reduce_window',
        'stablehlo.reduce_precision',
    }
    
    CONVERT_OPS = {
        'stablehlo.convert', 'stablehlo.bitcast_convert',
    }
    
    OTHER_OPS = {
        'stablehlo.constant', 'stablehlo.iota', 'stablehlo.rng',
        'stablehlo.custom_call', 'stablehlo.tuple', 'stablehlo.get_tuple_element',
    }
    
    def __init__(self):
        """Initialize the non-compute operations store."""
        self.ops: List[OpInfo] = []
        self.ops_by_category: dict = {
            'elementwise': [],
            'activation': [],
            'reshape': [],
            'reduction': [],
            'convert': [],
            'other': [],
        }
    
    def add_op(self, op: OpInfo) -> None:
        """
        Add a non-compute operation to the store.
        
        Args:
            op: OpInfo object representing the StableHLO operation
        """
        self.ops.append(op)
        
        # Categorize the operation
        op_name_lower = op.op_name.lower()
        
        if op.op_name in self.ELEMENTWISE_OPS or any(e in op_name_lower for e in ['add', 'subtract', 'multiply', 'divide', 'maximum', 'minimum']):
            self.ops_by_category['elementwise'].append(op)
        elif op.op_name in self.ACTIVATION_OPS or 'logistic' in op_name_lower or 'tanh' in op_name_lower:
            self.ops_by_category['activation'].append(op)
        elif op.op_name in self.RESHAPE_OPS or any(r in op_name_lower for r in ['reshape', 'transpose', 'broadcast', 'slice', 'gather', 'scatter', 'pad']):
            self.ops_by_category['reshape'].append(op)
        elif op.op_name in self.REDUCTION_OPS or 'reduce' in op_name_lower:
            self.ops_by_category['reduction'].append(op)
        elif op.op_name in self.CONVERT_OPS or 'convert' in op_name_lower:
            self.ops_by_category['convert'].append(op)
        else:
            self.ops_by_category['other'].append(op)
    
    def get_ops(self) -> List[OpInfo]:
        """Return all stored operations."""
        return self.ops
    
    def get_ops_by_category(self, category: str) -> List[OpInfo]:
        """
        Get operations by category.
        
        Args:
            category: One of 'elementwise', 'activation', 'reshape', 'reduction', 'convert', 'other'
            
        Returns:
            List of OpInfo objects in the specified category
        """
        return self.ops_by_category.get(category, [])
    
    def get_op_count(self) -> int:
        """Return total number of stored operations."""
        return len(self.ops)
    
    def get_category_counts(self) -> dict:
        """Return counts of operations by category."""
        return {cat: len(ops) for cat, ops in self.ops_by_category.items()}
    
    def get_summary(self) -> str:
        """Return a summary string of stored operations."""
        lines = [f"Non-Compute Operations Store: {len(self.ops)} total operations"]
        for cat, ops in self.ops_by_category.items():
            if ops:
                lines.append(f"  {cat}: {len(ops)} ops")
                # Show first few op names
                op_names = list(set(op.op_name for op in ops[:5]))
                lines.append(f"    Examples: {', '.join(op_names)}")
        return '\n'.join(lines)
    
    def to_jsonable(self) -> dict:
        """Convert the store to a JSON-serializable dictionary."""
        return {
            'total_count': len(self.ops),
            'category_counts': self.get_category_counts(),
            'operations': [op.to_jsonable() for op in self.ops],
        }
    
    def __len__(self) -> int:
        return len(self.ops)
    
    def __iter__(self):
        return iter(self.ops)
    
    def __repr__(self) -> str:
        return f"NonComputeOpsStore({len(self.ops)} operations)"


class NonComputeLatencyPredictor:
    """
    Predicts latency for non-compute StableHLO operations using pre-trained models.
    
    This class loads pre-trained sklearn models for elementwise operations
    and predicts their execution latency based on input tensor shapes.
    
    Available models are stored as .pkl files in the model directory.
    """
    
    # Mapping from StableHLO op names to model file names
    OP_NAME_TO_MODEL = {
        'stablehlo.add': 'add',
        'stablehlo.subtract': 'subtract',
        'stablehlo.multiply': 'multiply',
        'stablehlo.maximum': 'maximum',
        'stablehlo.minimum': 'minimum',
    }
    
    def __init__(self, model_dir: str = None, generation: str = None, verbose: bool = True):
        """
        Initialize the latency predictor.

        Args:
            model_dir: Directory containing pre-trained model .pkl files.
                       If None, the directory is resolved from `generation`.
            generation: TPU generation (e.g. 'TPUv4', 'TPUv6e'); selects the
                        per-generation subdir under scalesim/model/ (tpuv4, tpuv6e,
                        ...). Ignored if `model_dir` is given. Falls back to tpuv4
                        if the requested generation's dir does not exist.
            verbose: Whether to print loading progress
        """
        self.verbose = verbose

        # Default model directory
        if model_dir is None:
            model_dir = self._resolve_model_dir(generation, verbose)
        else:
            model_dir = Path(model_dir)
        
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}  # op_name -> model
        self.available_ops: List[str] = []
        
        self._load_available_models()

    # TimeLinearModel config value -> per-generation model subdir name
    _GENERATION_DIRS = {
        "tpuv4": "tpuv4", "tpuv5e": "tpuv5e", "tpuv6e": "tpuv6e",
    }

    @staticmethod
    def _resolve_model_dir(generation: str, verbose: bool = True) -> Path:
        """Map a TPU generation (e.g. 'TPUv6e') to its scalesim/model/<gen> dir.
        Falls back to tpuv4 when generation is unknown/None or its dir is absent."""
        base = Path(__file__).parent / "model"
        default = base / "tpuv4"
        if not generation:
            return default
        sub = NonComputeLatencyPredictor._GENERATION_DIRS.get(generation.lower())
        if sub is None:
            return default
        cand = base / sub
        if not cand.exists():
            if verbose:
                print(f"Warning: no op models for {generation} at {cand}; "
                      f"falling back to {default}")
            return default
        return cand

    def _load_available_models(self) -> None:
        """Load all available models from the model directory."""
        if not self.model_dir.exists():
            if self.verbose:
                print(f"Warning: Model directory not found: {self.model_dir}")
            return
        
        if self.verbose:
            print(f"Loading latency prediction models from: {self.model_dir}")
        
        for pkl_file in self.model_dir.glob("*.pkl"):
            op_name = pkl_file.stem  # e.g., "add" from "add.pkl"
            try:
                with open(pkl_file, "rb") as f:
                    model_data = pickle.load(f)
                
                # Handle both raw model and dict format
                if isinstance(model_data, dict) and "model" in model_data:
                    self.models[op_name] = model_data["model"]
                else:
                    self.models[op_name] = model_data
                
                self.available_ops.append(op_name)
                
                if self.verbose:
                    print(f"  Loaded model: {op_name}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load {pkl_file}: {e}")
        
        if self.verbose:
            print(f"Loaded {len(self.models)} latency prediction models")
    
    def _make_features(self, shapes: List[Tuple[int, int, int]]) -> pd.DataFrame:
        """
        Create feature dataframe for model prediction.
        
        Args:
            shapes: List of (d0, d1, d2) shape tuples
            
        Returns:
            DataFrame with features: d0, d1, d2, size, log2_size
        """
        df = pd.DataFrame({
            "d0": [s[0] for s in shapes],
            "d1": [s[1] for s in shapes],
            "d2": [s[2] for s in shapes],
        })
        df["size"] = df["d0"] * df["d1"] * df["d2"]
        df["log2_size"] = np.log2(df["size"].clip(lower=1)).astype(np.float32)
        return df
    
    def _opinfo_to_shape(self, op: 'OpInfo') -> Tuple[int, int, int]:
        """
        Extract a 3D shape from an OpInfo object.
        
        Pads shapes with 1s to make them 3D.
        
        Args:
            op: OpInfo object
            
        Returns:
            Tuple (d0, d1, d2) representing the shape
        """
        if not op.input_types:
            return (1, 1, 1)
        
        # Use the first input's shape
        shape, _ = op.input_types[0]
        
        # Pad to 3D
        if len(shape) == 0:
            return (1, 1, 1)
        elif len(shape) == 1:
            return (shape[0], 1, 1)
        elif len(shape) == 2:
            return (shape[0], shape[1], 1)
        elif len(shape) == 3:
            return (shape[0], shape[1], shape[2])
        else:
            # For higher dimensions, flatten trailing dims
            d0 = shape[0]
            d1 = shape[1]
            d2 = int(np.prod(shape[2:]))
            return (d0, d1, d2)
    
    def _get_model_for_op(self, op_name: str) -> Optional[Any]:
        """
        Get the appropriate model for a StableHLO operation.
        
        Args:
            op_name: Full StableHLO operation name (e.g., "stablehlo.add")
            
        Returns:
            Model if available, None otherwise
        """
        # Check direct mapping
        if op_name in self.OP_NAME_TO_MODEL:
            model_name = self.OP_NAME_TO_MODEL[op_name]
            return self.models.get(model_name)
        
        # Try extracting op name from stablehlo prefix
        if op_name.startswith("stablehlo."):
            base_name = op_name.split(".")[-1]
            if base_name in self.models:
                return self.models[base_name]
        
        return None
    
    def predict_op_latency(self, op: 'OpInfo') -> Optional[float]:
        """
        Predict latency for a single operation.
        
        Args:
            op: OpInfo object representing the operation
            
        Returns:
            Predicted latency in microseconds, or None if model not available
        """
        model = self._get_model_for_op(op.op_name)
        if model is None:
            return None
        
        shape = self._opinfo_to_shape(op)
        features = self._make_features([shape])
        
        prediction = model.predict(features)[0]
        return float(prediction)
    
    def predict_batch(self, ops: List['OpInfo']) -> List[Dict[str, Any]]:
        """
        Predict latencies for a batch of operations.
        
        Args:
            ops: List of OpInfo objects
            
        Returns:
            List of result dictionaries with op info and predictions
        """
        results = []
        
        for idx, op in enumerate(ops):
            shape = self._opinfo_to_shape(op)
            model = self._get_model_for_op(op.op_name)
            
            result = {
                "idx": idx,
                "op_name": op.op_name,
                "shape": shape,
                "size": shape[0] * shape[1] * shape[2],
                "has_model": model is not None,
                "predicted_latency": None,
            }
            
            if model is not None:
                features = self._make_features([shape])
                result["predicted_latency"] = float(model.predict(features)[0])
            
            results.append(result)
        
        return results
    
    def predict_from_store(self, store: NonComputeOpsStore) -> List[Dict[str, Any]]:
        """
        Predict latencies for all operations in a NonComputeOpsStore.
        
        Args:
            store: NonComputeOpsStore containing operations to predict
            
        Returns:
            List of result dictionaries with op info and predictions
        """
        return self.predict_batch(store.get_ops())
    
    def write_predictions_to_file(
        self, 
        results: List[Dict[str, Any]], 
        output_path: str,
        format: str = "csv"
    ) -> None:
        """
        Write prediction results to a file.
        
        Args:
            results: List of result dictionaries from predict_batch
            output_path: Path to output file
            format: Output format ("csv" or "json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "csv":
            df = pd.DataFrame(results)
            # Expand shape tuple to separate columns
            df["dim_0"] = df["shape"].apply(lambda x: x[0])
            df["dim_1"] = df["shape"].apply(lambda x: x[1])
            df["dim_2"] = df["shape"].apply(lambda x: x[2])
            df = df.drop(columns=["shape"])
            
            # Reorder columns
            cols = ["idx", "op_name", "dim_0", "dim_1", "dim_2", "size", 
                    "has_model", "predicted_latency"]
            df = df[cols]
            
            df.to_csv(output_path, index=False)
            
        elif format.lower() == "json":
            # Convert shapes to lists for JSON serialization
            json_results = []
            for r in results:
                jr = r.copy()
                jr["shape"] = list(jr["shape"])
                json_results.append(jr)
            
            with open(output_path, "w") as f:
                json.dump(json_results, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'.")
        
        if self.verbose:
            print(f"Predictions written to: {output_path}")
    
    def predict_and_save(
        self, 
        store: NonComputeOpsStore, 
        output_path: str,
        format: str = "csv"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Predict latencies for a store and save to file.
        
        Args:
            store: NonComputeOpsStore containing operations
            output_path: Path to output file
            format: Output format ("csv" or "json")
            
        Returns:
            Tuple of (results list, summary statistics)
        """
        results = self.predict_from_store(store)
        self.write_predictions_to_file(results, output_path, format)
        
        # Calculate summary statistics
        predicted_ops = [r for r in results if r["has_model"]]
        unpredicted_ops = [r for r in results if not r["has_model"]]
        
        summary = {
            "total_ops": len(results),
            "predicted_ops": len(predicted_ops),
            "unpredicted_ops": len(unpredicted_ops),
            "total_predicted_latency": sum(r["predicted_latency"] for r in predicted_ops),
            "ops_by_type": {},
        }
        
        # Count ops by type
        for r in results:
            op_name = r["op_name"]
            if op_name not in summary["ops_by_type"]:
                summary["ops_by_type"][op_name] = {"count": 0, "predicted": 0, "total_latency": 0}
            summary["ops_by_type"][op_name]["count"] += 1
            if r["has_model"]:
                summary["ops_by_type"][op_name]["predicted"] += 1
                summary["ops_by_type"][op_name]["total_latency"] += r["predicted_latency"]
        
        if self.verbose:
            print(f"\nPrediction Summary:")
            print(f"  Total operations:      {summary['total_ops']}")
            print(f"  With model available:  {summary['predicted_ops']}")
            print(f"  Without model:         {summary['unpredicted_ops']}")
            print(f"  Total predicted time:  {summary['total_predicted_latency']:.6f} μs")
            print(f"\n  Operations by type:")
            for op_name, stats in summary["ops_by_type"].items():
                model_status = "✓" if stats["predicted"] > 0 else "✗"
                latency_str = f"{stats['total_latency']:.6f} μs" if stats["predicted"] > 0 else "N/A"
                print(f"    {model_status} {op_name}: {stats['count']} ops, latency: {latency_str}")
        
        return results, summary
    
    def get_available_ops(self) -> List[str]:
        """Return list of operation names with available models."""
        return self.available_ops.copy()
    
    def is_op_supported(self, op_name: str) -> bool:
        """Check if a model is available for the given operation."""
        return self._get_model_for_op(op_name) is not None
    
    def __repr__(self) -> str:
        return f"NonComputeLatencyPredictor(models={self.available_ops})"


class StableHLOConverter:
    """
    Converter class that translates StableHLO operations to SCALE-Sim topology format.
    
    Supports:
    - stablehlo.convolution -> Conv topology format
    - stablehlo.dot_general -> GEMM topology format
    - stablehlo.dot -> GEMM topology format
    """
    
    def __init__(self, mlir_file: str, verbose: bool = True):
        """
        Initialize the converter with a StableHLO MLIR file.
        
        Args:
            mlir_file: Path to the .mlir file containing StableHLO operations
            verbose: Whether to print conversion progress
        """
        if not STABLEHLO_AVAILABLE:
            raise RuntimeError(
                "StableHLO parser is not available. Please install jax/jaxlib "
                "and ensure stablehlo_parse_min is accessible."
            )
        
        self.mlir_file = mlir_file
        self.verbose = verbose
        self.parser = StableHLOParser(mlir_path=str(mlir_file))
        self.ops = self.parser.get_ops_list()
        
        # Store for non-compute operations (not conv or gemm)
        self.non_compute_ops = NonComputeOpsStore()
        
        if self.verbose:
            print(f"Loaded {len(self.ops)} operations from {mlir_file}")
    
    def _convert_convolution_to_topology(self, op: OpInfo, op_idx: int, as_gemm: bool = False) -> Optional[List]:
        """
        Convert a StableHLO convolution operation to SCALE-Sim topology format.
        
        SCALE-Sim Conv format:
        [layer_name, ifmap_h, ifmap_w, filter_h, filter_w, num_ch, num_filt, stride_h, stride_w, N, M]
        
        SCALE-Sim GEMM format (for mixed workloads):
        [layer_name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
        where M = ofmap_h * ofmap_w, K = filter_h * filter_w * channels, N = num_filters
        
        Args:
            op: OpInfo object containing the convolution operation
            op_idx: Index of the operation for naming
            as_gemm: If True, convert to GEMM format for mixed workloads
            
        Returns:
            List in SCALE-Sim topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Convolution op {op_idx} has insufficient inputs/outputs")
            return None
        
        # Get input and filter shapes
        input_shape, _ = op.input_types[0]  # [batch, in_channels, height, width] or similar
        filter_shape, _ = op.input_types[1]  # [out_channels, in_channels, kh, kw] or similar
        output_shape, _ = op.output_types[0]
        
        # Parse dimension numbers if available
        dim_numbers = op.extra.get('dimension_numbers', {})
        
        if not dim_numbers:
            # Default assumption: [N, C, H, W] format for input, [OC, IC, KH, KW] for filter
            if len(input_shape) == 4 and len(filter_shape) == 4:
                batch = input_shape[0]
                in_channels = input_shape[1]
                ifmap_h = input_shape[2]
                ifmap_w = input_shape[3]
                
                out_channels = filter_shape[0]
                filter_h = filter_shape[2]
                filter_w = filter_shape[3]
            else:
                if self.verbose:
                    print(f"Warning: Unexpected convolution shapes at op {op_idx}")
                return None
        else:
            # Use dimension numbers to properly index
            lhs_dims = dim_numbers.get('lhs', {})
            rhs_dims = dim_numbers.get('rhs', {})
            
            # Extract from lhs (input)
            batch_dim = lhs_dims.get('batch_dimension', 0)
            feature_dim = lhs_dims.get('feature_dimension', 1)
            spatial_dims = [i for i in range(len(input_shape)) 
                           if i not in [batch_dim, feature_dim]]
            
            batch = input_shape[batch_dim]
            in_channels = input_shape[feature_dim]
            
            if len(spatial_dims) >= 2:
                ifmap_h = input_shape[spatial_dims[0]]
                ifmap_w = input_shape[spatial_dims[1]]
            elif len(spatial_dims) == 1:
                ifmap_h = input_shape[spatial_dims[0]]
                ifmap_w = 1
            else:
                ifmap_h = ifmap_w = 1
            
            # Extract from rhs (filter)
            out_feature_dim = rhs_dims.get('output_feature_dimension', 0)
            in_feature_dim = rhs_dims.get('input_feature_dimension', 1)
            filter_spatial_dims = [i for i in range(len(filter_shape))
                                   if i not in [out_feature_dim, in_feature_dim]]
            
            out_channels = filter_shape[out_feature_dim]
            
            if len(filter_spatial_dims) >= 2:
                filter_h = filter_shape[filter_spatial_dims[0]]
                filter_w = filter_shape[filter_spatial_dims[1]]
            elif len(filter_spatial_dims) == 1:
                filter_h = filter_shape[filter_spatial_dims[0]]
                filter_w = 1
            else:
                filter_h = filter_w = 1
        
        # Calculate stride (approximate from output shape)
        # output_h = (input_h - filter_h) / stride_h + 1
        if len(output_shape) >= 3:
            output_h = output_shape[-2] if len(output_shape) == 4 else output_shape[-1]
            stride_h = max(1, (ifmap_h - filter_h) // (output_h - 1)) if output_h > 1 else 1
            
            if len(output_shape) == 4:
                output_w = output_shape[-1]
                stride_w = max(1, (ifmap_w - filter_w) // (output_w - 1)) if output_w > 1 else 1
            else:
                stride_w = 1
        else:
            stride_h = stride_w = 1
        
        layer_name = f"conv_{op_idx}"
        
        # Sparsity ratio (default 1:1, meaning no sparsity)
        N_sparse, M_sparse = 1, 1
        
        if as_gemm:
            # Convert to GEMM format for mixed workloads
            # M = ofmap_h * ofmap_w (number of output pixels per channel)
            # K = filter_h * filter_w * in_channels (filter volume)
            # N = out_channels (number of filters)
            M = int(output_h * output_w) if len(output_shape) >= 3 else 1
            K = int(filter_h * filter_w * in_channels)
            N = int(out_channels)
            
            topology_entry = [
                layer_name,
                M,
                K,
                1,
                K,
                1,
                N,
                1,
                1,
                N_sparse,
                M_sparse
            ]
            
            if self.verbose:
                print(f"  Converted {op.op_name} -> {layer_name} (as GEMM): "
                      f"M={M}, N={N}, K={K} "
                      f"(from conv: ifmap={ifmap_h}x{ifmap_w}, filter={filter_h}x{filter_w}, ch={in_channels}→{out_channels})")
        else:
            # Standard convolution format
            topology_entry = [
                layer_name,
                int(ifmap_h),
                int(ifmap_w),
                int(filter_h),
                int(filter_w),
                int(in_channels),
                int(out_channels),
                int(stride_h),
                int(stride_w),
                N_sparse,
                M_sparse
            ]
            
            if self.verbose:
                print(f"  Converted {op.op_name} -> {layer_name}: "
                      f"ifmap={ifmap_h}x{ifmap_w}, filter={filter_h}x{filter_w}, "
                      f"ch={in_channels}, filters={out_channels}, stride={stride_h}x{stride_w}")
        
        return topology_entry
    
    def _convert_dot_general_to_gemm(self, op: OpInfo, op_idx: int) -> Optional[List]:
        """
        Convert a StableHLO dot_general operation to SCALE-Sim GEMM format.
        
        SCALE-Sim GEMM format:
        [layer_name, M, K, 1, K, 1, N, 1, 1, N_sparsity, M_sparsity]
        
        Simplified format for topology file (MNK):
        [layer_name, M, N, K, N_sparsity:M_sparsity]
        
        Args:
            op: OpInfo object containing the dot_general operation
            op_idx: Index of the operation for naming
            
        Returns:
            List in SCALE-Sim GEMM topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Dot_general op {op_idx} has insufficient inputs/outputs")
            return None
        
        lhs_shape, _ = op.input_types[0]
        rhs_shape, _ = op.input_types[1]
        output_shape, _ = op.output_types[0]
        
        # Get dimension information from extra
        dims = op.extra.get('dims', {})
        lhs_contracting = dims.get('lhs', [])
        rhs_contracting = dims.get('rhs', [])
        batch_dims = dims.get('batch', [])
        
        # Calculate M, N, K for GEMM
        # M: non-contracting dimensions from lhs
        # N: non-contracting dimensions from rhs
        # K: contracting dimensions
        
        # Calculate K (contracting dimension size)
        K = 1
        for dim_idx in lhs_contracting:
            if dim_idx < len(lhs_shape):
                K *= lhs_shape[dim_idx]
        
        # Calculate M (batch * non-contracting from lhs)
        lhs_batch_size = 1
        for batch_pair in batch_dims:
            lhs_batch_idx = batch_pair[0]
            if lhs_batch_idx < len(lhs_shape):
                lhs_batch_size *= lhs_shape[lhs_batch_idx]
        
        M = lhs_batch_size
        for dim_idx in range(len(lhs_shape)):
            if dim_idx not in lhs_contracting and dim_idx not in [b[0] for b in batch_dims]:
                M *= lhs_shape[dim_idx]
        
        # Calculate N (batch * non-contracting from rhs)
        rhs_batch_size = 1
        for batch_pair in batch_dims:
            rhs_batch_idx = batch_pair[1]
            if rhs_batch_idx < len(rhs_shape):
                rhs_batch_size *= rhs_shape[rhs_batch_idx]
        
        N = rhs_batch_size
        for dim_idx in range(len(rhs_shape)):
            if dim_idx not in rhs_contracting and dim_idx not in [b[1] for b in batch_dims]:
                N *= rhs_shape[dim_idx]
        
        # Ensure we have valid dimensions
        M = max(1, int(M))
        N = max(1, int(N))
        K = max(1, int(K))
        
        layer_name = f"gemm_{op_idx}"
        
        # Sparsity ratio (default 1:1)
        N_sparse, M_sparse = 1, 1
        
        # Return in the internal format used by SCALE-Sim's GEMM loader
        # Format: [name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
        topology_entry = [
            layer_name,
            M,
            K,
            1,
            K,
            1,
            N,
            1,
            1,
            N_sparse,
            M_sparse
        ]
        
        if self.verbose:
            print(f"  Converted {op.op_name} -> {layer_name}: M={M}, N={N}, K={K}")
        
        return topology_entry
    
    def _convert_dot_to_gemm(self, op: OpInfo, op_idx: int) -> Optional[List]:
        """
        Convert a StableHLO dot operation to SCALE-Sim GEMM format.
        
        For basic dot (matrix multiplication), assume:
        - lhs: [M, K]
        - rhs: [K, N]
        - output: [M, N]
        
        Args:
            op: OpInfo object containing the dot operation
            op_idx: Index of the operation for naming
            
        Returns:
            List in SCALE-Sim GEMM topology format, or None if conversion fails
        """
        if len(op.input_types) < 2 or len(op.output_types) < 1:
            if self.verbose:
                print(f"Warning: Dot op {op_idx} has insufficient inputs/outputs")
            return None
        
        lhs_shape, _ = op.input_types[0]
        rhs_shape, _ = op.input_types[1]
        
        # For simple matrix multiplication: [M, K] x [K, N] -> [M, N]
        if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
            M = int(lhs_shape[-2]) if len(lhs_shape) >= 2 else 1
            K = int(lhs_shape[-1])
            N = int(rhs_shape[-1])
            
            # Handle batch dimensions
            batch_size = 1
            for dim in lhs_shape[:-2]:
                batch_size *= dim
            M *= batch_size
        else:
            if self.verbose:
                print(f"Warning: Unexpected dot shapes at op {op_idx}")
            return None
        
        layer_name = f"gemm_{op_idx}"
        
        # Sparsity ratio (default 1:1)
        N_sparse, M_sparse = 1, 1
        
        topology_entry = [
            layer_name,
            M,
            K,
            1,
            K,
            1,
            N,
            1,
            1,
            N_sparse,
            M_sparse
        ]
        
        if self.verbose:
            print(f"  Converted {op.op_name} -> {layer_name}: M={M}, N={N}, K={K}")
        
        return topology_entry
    
    def convert_to_topology(self) -> Tuple[List[List], str]:
        """
        Convert all StableHLO operations to SCALE-Sim topology format.
        
        For mixed workloads (conv + matmul), all operations are converted to GEMM format
        since SCALE-Sim can handle convolutions as GEMM operations (im2col).
        
        Returns:
            Tuple of (topology_entries, input_type) where:
            - topology_entries: List of topology entries
            - input_type: Either "conv" or "gemm" indicating the format used
        """
        # First pass: count operation types to determine format
        conv_count = 0
        gemm_count = 0
        
        for op in self.ops:
            if "convolution" in op.op_name.lower():
                conv_count += 1
            elif "dot_general" in op.op_name.lower() or op.op_name == "stablehlo.dot":
                gemm_count += 1
        
        # Determine format: if we have both conv and gemm, use gemm format for all
        use_gemm_format = (gemm_count > 0)
        input_type = "gemm" if use_gemm_format else "conv"
        
        if self.verbose:
            print(f"\nConverting {len(self.ops)} operations to SCALE-Sim topology...")
            if conv_count > 0 and gemm_count > 0:
                print(f"  → Detected mixed workload ({conv_count} conv + {gemm_count} matmul)")
                print(f"  → Using GEMM format for all layers (convs converted to GEMM via im2col)")
        
        # Second pass: convert operations with the determined format
        topology_entries = []
        
        for idx, op in enumerate(self.ops):
            entry = None
            
            if "convolution" in op.op_name.lower():
                entry = self._convert_convolution_to_topology(op, idx, as_gemm=use_gemm_format)
                if entry:
                    pass  # Already counted
            
            elif "dot_general" in op.op_name.lower():
                entry = self._convert_dot_general_to_gemm(op, idx)
                if entry:
                    pass  # Already counted
            
            elif op.op_name == "stablehlo.dot":
                entry = self._convert_dot_to_gemm(op, idx)
                if entry:
                    pass  # Already counted
            
            else:
                # Store non-compute operations (not conv or gemm)
                self.non_compute_ops.add_op(op)
                if self.verbose:
                    print(f"  Stored non-compute op: {op.op_name}")
            
            if entry:
                topology_entries.append(entry)
        
        if self.verbose:
            print(f"\nConversion complete:")
            print(f"  Total operations: {len(self.ops)}")
            print(f"  Converted convolutions: {conv_count}")
            print(f"  Converted GEMMs: {gemm_count}")
            print(f"  Non-compute operations stored: {len(self.non_compute_ops)}")
            print(f"  Output format: {input_type}")
        
        return topology_entries, input_type
    
    def build_op_table(self, model_dir: str = None, generation: str = None) -> List[Dict[str, Any]]:
        """
        Build a program-ordered table of every op (compute + non-compute) for the
        unified TIME_REPORT. One row per op, in MLIR program order:

          op_id    : global program index (stable, orders the report)
          kind     : 'compute' (dot_general/convolution/dot) or 'noncompute'
          op       : short op name (e.g. 'transpose')
          stablehlo: short signature 'op in_shapes->out_shape dtype' (no attrs;
                     the latency models are shape-only, so attrs are omitted)
          layer    : compute-layer index == COMPUTE_REPORT LayerID (None/N/A for
                     non-compute) -- lets a row be matched to the cycle report
          time_us  : non-compute predicted latency (None if no model); compute
                     times are filled in later from the simulator's per-layer time
          modeled  : whether a latency was produced

        Call after convert_to_topology(). Mirrors that method's compute/non-compute
        split exactly, so layer indexing matches the topology / COMPUTE_REPORT.
        """
        predictor = NonComputeLatencyPredictor(model_dir=model_dir, generation=generation, verbose=False)
        table: List[Dict[str, Any]] = []
        layer = 0
        for gidx, op in enumerate(self.ops):
            nm = op.op_name.lower()
            is_compute = ("convolution" in nm or "dot_general" in nm
                          or op.op_name == "stablehlo.dot")
            row = {"op_id": gidx, "op": op.op_name.split(".")[-1],
                   "stablehlo": _short_op_sig(op)}
            if is_compute:
                row.update(kind="compute", layer=layer, time_us=None, modeled=True)
                layer += 1
            else:
                t = predictor.predict_op_latency(op)
                row.update(kind="noncompute", layer=None, time_us=t,
                           modeled=t is not None)
            table.append(row)
        return table

    def get_non_compute_ops(self) -> NonComputeOpsStore:
        """
        Get the store containing all non-compute operations.
        
        Note: This is populated after calling convert_to_topology().
        
        Returns:
            NonComputeOpsStore containing all non-conv and non-gemm operations
        """
        return self.non_compute_ops
    
    def get_non_compute_ops_list(self) -> List[OpInfo]:
        """
        Get a list of all non-compute OpInfo objects.
        
        Note: This is populated after calling convert_to_topology().
        
        Returns:
            List of OpInfo objects for non-conv and non-gemm operations
        """
        return self.non_compute_ops.get_ops()
    
    def predict_non_compute_latencies(
        self, 
        output_path: str = None,
        model_dir: str = None,
        format: str = "csv"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Predict latencies for all non-compute operations using pre-trained models.
        
        Note: Call convert_to_topology() first to populate non_compute_ops.
        
        Args:
            output_path: Path to save predictions (optional). If None, no file is written.
            model_dir: Directory containing model .pkl files. If None, uses default.
            format: Output format ("csv" or "json")
            
        Returns:
            Tuple of (results list, summary statistics)
        """
        if len(self.non_compute_ops) == 0:
            if self.verbose:
                print("Warning: No non-compute operations found. "
                      "Did you call convert_to_topology() first?")
            return [], {"total_ops": 0, "predicted_ops": 0, "unpredicted_ops": 0}
        
        predictor = NonComputeLatencyPredictor(model_dir=model_dir, verbose=self.verbose)
        
        if output_path:
            return predictor.predict_and_save(self.non_compute_ops, output_path, format)
        else:
            results = predictor.predict_from_store(self.non_compute_ops)
            # Calculate summary
            predicted_ops = [r for r in results if r["has_model"]]
            summary = {
                "total_ops": len(results),
                "predicted_ops": len(predicted_ops),
                "unpredicted_ops": len(results) - len(predicted_ops),
                "total_predicted_latency": sum(r["predicted_latency"] for r in predicted_ops),
            }
            return results, summary
    
    def save_to_csv(self, output_path: str, input_type: str = None) -> str:
        """
        Convert StableHLO operations and save to a SCALE-Sim topology CSV file.
        
        Args:
            output_path: Path where the CSV file should be saved
            input_type: Override the auto-detected input type ("conv" or "gemm")
            
        Returns:
            The input type that was used
        """
        topology_entries, detected_type = self.convert_to_topology()
        
        # Use provided input_type or the detected one
        final_input_type = input_type if input_type else detected_type
        
        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV file
        with open(output_path, 'w') as f:
            if final_input_type == "conv":
                # Conv format header
                f.write("Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width, "
                       "Channels, Num Filter, Strides,\n")
                
                for entry in topology_entries:
                    # entry: [name, ifmap_h, ifmap_w, filter_h, filter_w, ch, num_filt, stride_h, stride_w, N, M]
                    # CSV format needs stride only once (stride_h is used)
                    line = f"{entry[0]}, {entry[1]}, {entry[2]}, {entry[3]}, {entry[4]}, " \
                           f"{entry[5]}, {entry[6]}, {entry[7]},\n"
                    f.write(line)
            else:
                # GEMM format header
                f.write("Layer,M,N,K,\n")
                
                for entry in topology_entries:
                    # entry: [name, M, K, 1, K, 1, N, 1, 1, N_sparse, M_sparse]
                    # Extract M, N, K from the entry
                    name = entry[0]
                    M = entry[1]
                    N = entry[6]
                    K = entry[2]
                    line = f"{name},{M},{N},{K},\n"
                    f.write(line)
        
        if self.verbose:
            print(f"\nTopology saved to: {output_path}")
            print(f"Format: {final_input_type}")
        
        return final_input_type


def _short_op_sig(op: 'OpInfo') -> str:
    """Compact, shapes-only signature: 'name in1·in2->out dtype' (no attributes,
    since the latency models are shape-only). E.g.
      'transpose 1x128x12x64->1x12x128x64 f32'
      'dot_general 128x768·768x2304->128x2304 f32'."""
    def shp(sig):
        dims, _ = sig
        return "x".join(str(d) for d in dims) if dims else "scalar"
    name = op.op_name.split(".")[-1]
    ins = "·".join(shp(s) for s in op.input_types) if op.input_types else ""
    out = shp(op.output_types[0]) if op.output_types else ""
    dt = (op.input_types[0][1] if op.input_types
          else (op.output_types[0][1] if op.output_types else ""))
    core = f"{ins}->{out}" if (ins and out) else (ins or out)
    return " ".join(p for p in (name, core, dt) if p)


def convert_stablehlo_to_topology(mlir_file: str, output_csv: str = None,
                                   verbose: bool = False) -> Tuple[str, str]:
    """
    Convenience function to convert a StableHLO MLIR file to SCALE-Sim topology CSV.
    
    Args:
        mlir_file: Path to the .mlir file
        output_csv: Path for the output CSV (if None, generates from input name)
        verbose: Whether to print conversion progress
        
    Returns:
        Tuple of (output_csv_path, input_type)
    """
    if not STABLEHLO_AVAILABLE:
        raise RuntimeError(
            "StableHLO parser is not available. Please install jax/jaxlib:\n"
            "  pip install jax jaxlib"
        )
    
    # Generate output path if not provided
    if output_csv is None:
        mlir_path = Path(mlir_file)
        output_csv = str(mlir_path.parent / f"{mlir_path.stem}_topology.csv")
    
    # Convert and save
    converter = StableHLOConverter(mlir_file, verbose=verbose)
    input_type = converter.save_to_csv(output_csv)
    
    return output_csv, input_type


def convert_mlir_if_needed(
    topology_file: str,
    inp_type: str,
    logpath: str,
    config_file: str = None
) -> Tuple[str, str, bool]:
    """
    Check if the topology file is a .mlir file and convert it if needed.

    This function is used by SCALE-Sim's main entry point to automatically
    detect and convert MLIR files to topology CSV format. It also automatically
    predicts latencies for non-compute operations and saves them to a time report.

    Args:
        topology_file: Path to the topology or MLIR file
        inp_type: Input type (conv/gemm/auto)
        logpath: Directory for logs and converted files
        config_file: Path to the SCALE-Sim config; its TimeLinearModel key selects
                     the per-generation op-latency models (e.g. TPUv6e -> tpuv6e).

    Returns:
        Tuple of (topology_csv_path, input_type, is_converted)
        - topology_csv_path: Path to the topology CSV (converted or original)
        - input_type: Final input type to use (conv/gemm)
        - is_converted: True if conversion was performed
    """
    topology_path = Path(topology_file)
    
    # Check if it's a .mlir file
    if topology_path.suffix.lower() == '.mlir':
        if not STABLEHLO_AVAILABLE:
            print("ERROR: Cannot process .mlir files - StableHLO parser not available")
            print("Please install required dependencies:")
            print("  pip install jax jaxlib")
            import sys
            sys.exit(1)
        
        print(f"\nDetected StableHLO MLIR file: {topology_file}")
        print("Converting to SCALE-Sim topology format...\n")
        
        # Generate output CSV path in the log directory
        output_csv = Path(logpath) / f"{topology_path.stem}_converted_topology.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Create converter and convert
        converter = StableHLOConverter(str(topology_file), verbose=False)
        detected_type = converter.save_to_csv(str(output_csv))
        
        # Use detected type if input type is auto
        if inp_type == "auto":
            final_type = detected_type
        else:
            final_type = inp_type
        
        print(f"Converted topology saved to: {output_csv}")
        
        # Build the program-ordered op table (compute + non-compute, with
        # non-compute latencies predicted) and stash it for the post-run combiner,
        # which merges it with the simulator's per-layer compute times into the
        # single unified TIME_REPORT.csv (see scalesim/total_time_report.py).
        # Resolve the TPU generation from the config so the op-latency models match
        # the configured TimeLinearModel (e.g. TPUv6e -> scalesim/model/tpuv6e/).
        generation = None
        if config_file:
            try:
                from scalesim.scale_config import scale_config
                cfg = scale_config()
                cfg.read_conf_file(config_file)
                gen = cfg.get_time_linear_model()
                if gen and gen not in ("None", "Default"):
                    generation = gen
            except Exception as e:
                print(f"Warning: could not read TimeLinearModel from config: {e}")

        op_table = converter.build_op_table(generation=generation)
        op_table_path = Path(logpath) / f"{topology_path.stem}_op_table.json"
        with open(op_table_path, "w") as f:
            json.dump(op_table, f)
        print(f"Op table saved to: {op_table_path}")

        return str(output_csv), final_type, True
    
    # Not a MLIR file, return as-is
    return topology_file, inp_type, False


def _write_non_compute_time_report(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write non-compute latency predictions to a simple time report CSV.
    
    Format matches TIME_REPORT.csv:
    LayerID, Time (us),
    
    Args:
        results: List of prediction result dicts from NonComputeLatencyPredictor
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("LayerID, Time (us),\n")
        for r in results:
            layer_id = r["idx"]
            # Use predicted latency if available, otherwise N/A
            if r["predicted_latency"] is not None:
                f.write(f"{layer_id}, {r['predicted_latency']},\n")
            else:
                f.write(f"{layer_id}, N/A,\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert StableHLO MLIR to SCALE-Sim topology CSV")
    parser.add_argument("mlir_file", type=str, help="Path to the .mlir file")
    parser.add_argument("-o", "--output", type=str, default=None,
                       help="Output CSV file path (default: <input>_topology.csv)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    try:
        output_csv, input_type = convert_stablehlo_to_topology(
            args.mlir_file,
            args.output,
            verbose=not args.quiet
        )
        print(f"\nSuccess! Use with SCALE-Sim:")
        print(f"  python3 -m scalesim.scale -t {output_csv} -i {input_type} -c <config_file>")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

