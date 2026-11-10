"""
This file is the main script for running SCALE-Sim with the given topology and configuration files.
It handles argument parsing and execution.
"""

import argparse
import os

from scalesim.scale_sim import scalesim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="./topologies/conv_nets/test.csv",
                        help="Path to the topology file"
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
                        default="conv",
                        help="Type of input topology, gemm: MNK, conv: conv"
                        )
    parser.add_argument('-s', metavar='save trace', type=str,
                        default="Y",
                        help="Save Trace: (Y/N)"
                        )
    parser.add_argument('-m', metavar='hardware metrics', type=str,
                        default="",
                        help="Path to NCU hardware metrics CSV for comparison report"
                        )

    args = parser.parse_args()
    topology    = args.t
    layout      = args.l
    config      = args.c
    logpath     = args.p
    inp_type    = args.i
    save_trace  = args.s
    ncu_metrics = args.m

    GEMM_INPUT = inp_type == 'gemm'
    save_space = save_trace != 'Y'

    s = scalesim(save_disk_space=save_space,
                 verbose=True,
                 config=config,
                 topology=topology,
                 layout=layout,
                 input_type_gemm=GEMM_INPUT,
                 ncu_metrics=ncu_metrics
                 )
    s.run_scale(top_path=logpath)

    if ncu_metrics:
        from scalesim.compare_metrics import run_comparison

        # The simulator writes reports to <logpath>/<run_name>/
        run_name    = s.config.get_run_name()
        simdir      = os.path.join(logpath, run_name)
        report_path = os.path.join(simdir, "comparison_report.txt")

        run_comparison(ncu_path=ncu_metrics,
                       scalesim_dir=simdir,
                       report_path=report_path)
