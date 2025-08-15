from scalesim.scale_sim import scalesim
import os

absolute_dir_path = os.path.dirname(os.path.realpath(__file__))

topology = absolute_dir_path + "/../topologies/MLperf_tiny/vww.csv"
config = absolute_dir_path + "/../configs/edgeTPU.cfg"
logpath = absolute_dir_path + "/../edgeTPU_experiment_results"
# inp_type = absolute_dir_path + "/conv"

# make sure logpath exists
if not os.path.exists(logpath):
    os.makedirs(logpath)

# gemm_input = False
# if inp_type == 'gemm':
#     gemm_input = True


s = scalesim(save_disk_space=True, verbose=True,
                config=config,
                topology=topology,
                # input_type_gemm=gemm_input
                )
s.run_scale(top_path=logpath)