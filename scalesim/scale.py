"""
This file is the main script for running SCALE-Sim with the given topology and configuration files.
It handles argument parsing and execution.

Now supports StableHLO MLIR files as input with non-compute operation latency prediction!
"""

import argparse
import os

from scalesim.scale_sim import scalesim

# Import StableHLO converter if available
try:
    from scalesim.stablehlo_converter import convert_mlir_if_needed
    STABLEHLO_AVAILABLE = True
except ImportError:
    STABLEHLO_AVAILABLE = False
    # Fallback if converter not available
    def convert_mlir_if_needed(topology_file, inp_type, logpath):
        return topology_file, inp_type, False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SCALE-Sim: Systolic CNN Accelerator Simulator\n"
                    "Supports StableHLO MLIR files as input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic simulation with CSV topology
  python3 -m scalesim.scale -t topology.csv -c config.cfg

  # Simulate from StableHLO MLIR file (auto-converts and predicts non-compute latencies)
  python3 -m scalesim.scale -t model.mlir -c config.cfg
"""
    )
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="./topologies/conv_nets/test.csv",
                        help="Path to the topology file (.csv or .mlir)"
                        )
    parser.add_argument('-l', metavar='Layout file', type=str,
                        default="./layouts/conv_nets/test.csv",
                        help="Path to the layout file"
                        )
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="./configs/scale.cfg",
                        help="Path to the config file"
                        )
    parser.add_argument('-p', metavar='log dir', type=str,
                        default="./results/",
                        help="Path to log dir"
                        )
    parser.add_argument('-i', metavar='input type', type=str,
                        default="auto",
                        help="Type of input topology: conv, gemm, or auto (default: auto)"
                        )
    parser.add_argument('-s', metavar='save trace', type=str,
                        default="Y",
                        help="Save Trace: (Y/N)"
                        )
    parser.add_argument('-b', '--bypass', action='store_true',
                        help="Bypass the cycle-accurate compute simulation and use "
                             "the closed-form analytical cycle model (fast; emits "
                             "COMPUTE_REPORT.csv + TIME_REPORT.csv, no traces)"
                        )

    args = parser.parse_args()
    topology = args.t
    layout = args.l
    config = args.c
    logpath = args.p
    inp_type = args.i.lower()
    save_trace = args.s
    bypass_compute = args.bypass

    # Convert MLIR file if needed (automatically handles non-compute operations)
    topology, inp_type, was_converted = convert_mlir_if_needed(
        topology, 
        inp_type, 
        logpath
    )
    
    # Determine input type
    GEMM_INPUT = False
    if inp_type == 'gemm':
        GEMM_INPUT = True
    elif inp_type == "auto":
        # Default to conv if not specified
        GEMM_INPUT = False
        inp_type = "conv"
    
    if save_trace == 'Y':
        save_space = False
    else:
        save_space = True
   

    s = scalesim(save_disk_space=False,
                 verbose=True,
                 config=config,
                 topology=topology,
                 layout=layout,
                 input_type_gemm=GEMM_INPUT,
                 bypass_compute=bypass_compute
                 )
    s.run_scale(top_path=logpath)

    # Consolidate compute + non-compute into one unified TIME_REPORT.csv in the
    # run's inner directory (no-op for plain CSV-topology runs).
    try:
        from scalesim.total_time_report import write_total_time_report
        run_dir = os.path.join(logpath, s.config.get_run_name())
        if write_total_time_report(logpath, run_dir):
            print(f"Unified time report: {os.path.join(run_dir, 'TIME_REPORT.csv')}")
    except Exception as e:
        print(f"Warning: could not write unified time report: {e}")
