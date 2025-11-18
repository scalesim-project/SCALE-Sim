"""
This file is the main script for running SCALE-Sim with the given topology and configuration files.
It handles argument parsing and execution.
"""

import argparse

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

    args = parser.parse_args()
    topology = args.t
    layout = args.l
    config = args.c
    logpath = args.p
    inp_type = args.i
    save_trace = args.s

    GEMM_INPUT = False
    if inp_type == 'gemm':
        GEMM_INPUT = True
    
    if save_trace == 'Y':
        save_space = False
    else:
        save_space = True
   

    s = scalesim(save_disk_space=False,
                 verbose=True,
                 config=config,
                 topology=topology,
                 layout=layout,
                 input_type_gemm=GEMM_INPUT
                 )
    
    import time
    import os

    # start_time = time.time()

    s.run_scale(top_path=logpath)

    # end_time = time.time()
    # total_runtime = end_time - start_time

    # time_log_dir = "./results/bagel"
    # os.makedirs(time_log_dir, exist_ok=True)

    # time_log_file = os.path.join(time_log_dir, "time_result.log")

    # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    """with open(time_log_file, "a") as f:
        f.write(f"=== SCALE-Sim Execution Time Log ===\n")
        f.write(f"Timestamp: {current_time}\n")
        f.write(f"Topology: {topology}\n")
        f.write(f"Layout: {layout}\n") 
        f.write(f"Config: {config}\n")
        f.write(f"Input Type: {inp_type}\n")
        f.write(f"Save Trace: {save_trace}\n")
        f.write(f"Total Runtime: {total_runtime:.4f} seconds\n")
        f.write(f"Total Runtime: {total_runtime/60:.2f} minutes\n")
        f.write(f"{'='*50}\n\n")
    
    # 在终端也显示运行时间
    print(f"\n=== Execution Completed ===")
    print(f"Total Runtime: {total_runtime:.4f} seconds ({total_runtime/60:.2f} minutes)")
    print(f"Time log saved to: {time_log_file}")"""