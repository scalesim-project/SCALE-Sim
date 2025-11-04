"""
This file contains 'scalesim' class that provides a framework to run simulations, generate traces
and reports.
"""

import os
from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.layout_utils import layouts
from scalesim.simulator import simulator


class scalesim:
    """
    The top level class for the SCALE-Sim v2 simulator that provides methods for setting parameters,
    running sims, and generating results.
    """
    #
    def __init__(self,
                 save_disk_space=False,
                 verbose=True,
                 config='',
                 topology='',
                 layout='',
                 input_type_gemm=False):
        """
        __init__ method
        """
        # Data structures
        self.config = scale_config()
        self.topo = topologies()
        self.layout = layouts()

        # File paths
        self.config_file = ''
        self.topology_file = ''
        self.layout_file = ''
        self.top_path = ''
        # Member objects
        #self.runner = r.run_nets()
        self.runner = simulator()

        # Flags
        self.read_gemm_inputs = input_type_gemm
        self.save_space = save_disk_space
        self.verbose_flag = verbose
        self.run_done_flag = False
        self.logs_generated_flag = False

        self.set_params(config_filename=config, topology_filename=topology, layout_filename=layout)

    #
    def set_params(self,
                   config_filename='',
                   topology_filename='',
                   layout_filename=''):

        """
        Set or update the paths to the scalesim input files.
        """
        # First check if the user provided a valid topology file
        if topology_filename != '':
            if not os.path.exists(topology_filename):
                print("ERROR: scalesim.scale.py: Topology file not found")
                print("Input file:" + topology_filename)
                print('Exiting')
                exit()
            else:
                self.topology_file = topology_filename

        if not layout_filename == '':
            if not os.path.exists(layout_filename):
                print("ERROR: scalesim.scale.py: Layout file not found")
                print("Input file:" + layout_filename)
                print('Exiting')
                exit()
            else:
                self.layout_file = layout_filename

        if not os.path.exists(config_filename):
            print("ERROR: scalesim.scale.py: Config file not found")
            print("Input file:" + config_filename)
            print('Exiting')
            exit()
        else:
            self.config_file = config_filename

        # Parse config first
        self.config.read_conf_file(self.config_file)

        # Take the CLI topology over the one in config
        # If topology is not passed from CLI take the one from config
        if self.topology_file == '':
            self.topology_file = self.config.get_topology_path()
        else:
            self.config.set_topology_file(self.topology_file)

        if self.layout_file == '':
            self.layout_file = self.config.get_layout_path()
        else:
            self.config.set_layout_file(self.layout_file)

        # Parse the topology
        self.topo.load_arrays(topofile=self.topology_file, mnk_inputs=self.read_gemm_inputs)
        self.layout.load_arrays(layoutfile=self.layout_file, mnk_inputs=self.read_gemm_inputs)

        #num_layers = self.topo.get_num_layers()
        #self.config.scale_memory_maps(num_layers=num_layers)

    #
    def run_scale(self, top_path='.'):
        """
        Method to initialize the internal simulation objects and run scalesim once.
        """

        self.top_path = top_path
        save_trace = not self.save_space
        self.runner.set_params(
            config_obj=self.config,
            topo_obj=self.topo,
            layout_obj=self.layout,
            top_path=self.top_path,
            verbosity=self.verbose_flag,
            save_trace=save_trace
        )
        self.run_once()

    #
    def run_once(self):
        """
        Method to run the simulation once with preset config and topology objects.
        """

        if self.verbose_flag:
            self.print_run_configs()

        #save_trace = not self.save_space

        # TODO: Anand
        # TODO: This release
        # TODO: Call the class member functions
        #self.runner.run_net(
        #    config=self.config,
        #    topo=self.topo,
        #    top_path=self.top_path,
        #    save_trace=save_trace,
        #    verbosity=self.verbose_flag
        #)
        self.runner.run()
        self.run_done_flag = True

        #self.runner.generate_all_logs()
        self.logs_generated_flag = True

        if self.verbose_flag:
            print("************ SCALE SIM Run Complete ****************")

    #
    def print_run_configs(self):
        """
        Method to print the banner of input parameters for verbose scalesim runs.
        """
        df_string = "Output Stationary"
        df = self.config.get_dataflow()

        if df == 'ws':
            df_string = "Weight Stationary"
        elif df == 'is':
            df_string = "Input Stationary"

        print("====================================================")
        print("******************* SCALE SIM **********************")
        print("====================================================")

        arr_h, arr_w = self.config.get_array_dims()
        print("Array Size: \t" + str(arr_h) + "x" + str(arr_w))

        ifmap_kb, filter_kb, ofmap_kb = self.config.get_mem_sizes()
        print("SRAM IFMAP (kB): \t" + str(ifmap_kb))
        print("SRAM Filter (kB): \t" + str(filter_kb))
        print("SRAM OFMAP (kB): \t" + str(ofmap_kb))
        print("Dataflow: \t" + df_string)
        print("topology file path: \t" + self.config.get_topology_path())
        print("layout file path: \t" + self.config.get_layout_path())

        if self.config.use_user_dram_bandwidth():
            print("Bandwidth: \t" + self.config.get_bandwidths_as_string())
            print('Working in USE USER BANDWIDTH mode.')
        else:
            print('Working in ESTIMATE BANDWIDTH mode.')

        print("====================================================")

    #
    def get_total_cycles(self):
        """
        Method to get the total cycles (stalls + compute) for the workload once the simulation is
        completed.
        """
        me = 'scale.' + 'get_total_cycles()'
        if not self.run_done_flag:
            message = 'ERROR: ' + me
            message += ' : Cannot determine cycles. Run the simulation first'
            print(message)
            return

        return self.runner.get_total_cycles()
