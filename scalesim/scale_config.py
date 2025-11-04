"""
This file defines the 'scale_config' class responsible for all the configuration file related
activities such as parsing the config file, writing the parameters into a config file, updating the
parameters.
"""
import configparser as cp


class scale_config:
    """
    Class that handles the SCALE-Sim configuration files.
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.run_name = "scale_run"
        # Anand: ISSUE #2. Patch
        self.use_user_bandwidth = False

        self.array_rows = 4
        self.array_cols = 4
        self.ifmap_sz_kb = 256
        self.filter_sz_kb = 256
        self.ofmap_sz_kb = 128
        self.df = 'ws'
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000
        self.req_buf_sz_rd = 60
        self.req_buf_sz_wr = 60
        self.topofile = ""
        self.layoutfile = ""
        self.bandwidths = []
        self.valid_conf_flag = False
        self.num_bank = 1
        self.num_port = 2

        # Layout flags with default values
        self.using_ifmap_custom_layout = False
        self.ifmap_sram_bank_bandwidth = 10
        self.ifmap_sram_bank_num = 10
        self.ifmap_sram_bank_port = 2
        self.using_filter_custom_layout = False
        self.filter_sram_bank_bandwidth = 10
        self.filter_sram_bank_num = 10
        self.filter_sram_bank_port = 2

        self.valid_df_list = ['os', 'ws', 'is']

        self.sparsity_support = False
        self.sparsity_representation = ""
        # self.sparsity_N = 4
        # self.sparsity_M = 4
        self.sparsity_optimized_mapping = False
        self.sparsity_block_size = 4
        self.sparsity_rand_seed = 40
    
    # Sarbartha: Added ramulator based DRAM trace support
        self.use_ramulator_trace = False
    #
    def read_conf_file(self, conf_file_in):
        """
        Method to read the configuration file and extract all the archietctural knobs.
        """

        me = 'scale_config.' + 'read_conf_file()'

        config = cp.ConfigParser()
        config.read(conf_file_in)

        section = 'general'
        self.run_name = config.get(section, 'run_name')

        # Anand: ISSUE #2. Patch
        section = 'run_presets'
        bw_mode_string = config.get(section, 'InterfaceBandwidth')
        if bw_mode_string == 'USER':
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False
        else:
            message = 'ERROR: ' + me
            message += 'Use either USER or CALC in InterfaceBandwidth feild. Aborting!'
            return
        
        ramulator_on = config.get(section, 'UseRamulatorTrace')
        if ramulator_on == 'True':
            self.use_ramulator_trace = True
        else:
            self.use_ramulator_trace = False
        
        # TODO Sarbartha: Should be bw
        div_factor = 1
        
        section = 'architecture_presets'
        self.array_rows = int(config.get(section, 'ArrayHeight'))
        self.array_cols = int(config.get(section, 'ArrayWidth'))
        self.ifmap_sz_kb = int(config.get(section, 'ifmapsramszkB'))
        self.filter_sz_kb = int(config.get(section, 'filtersramszkB'))
        self.ofmap_sz_kb = int(config.get(section, 'ofmapsramszkB'))
        self.ifmap_offset = int(config.get(section, 'IfmapOffset'))
        self.filter_offset = int(config.get(section, 'FilterOffset'))
        self.ofmap_offset = int(config.get(section, 'OfmapOffset'))
        self.df = config.get(section, 'Dataflow')
        self.req_buf_sz_rd = int(config.get(section, 'ReadRequestBuffer')) // div_factor
        self.req_buf_sz_wr = int(config.get(section, 'WriteRequestBuffer')) // div_factor

        layout_section = 'layout'
        self.using_ifmap_custom_layout = config.getboolean(layout_section, 'IfmapCustomLayout')
        self.using_filter_custom_layout = config.getboolean(layout_section, 'FilterCustomLayout')
        self.ifmap_sram_bank_bandwidth = int(config.get(layout_section, 'IfmapSRAMBankBandwidth'))
        self.ifmap_sram_bank_num = int(config.get(layout_section, 'IfmapSRAMBankNum'))
        self.ifmap_sram_bank_port = int(config.get(layout_section, 'IfmapSRAMBankPort'))
        self.filter_sram_bank_bandwidth = int(config.get(layout_section, 'FilterSRAMBankBandwidth'))
        self.filter_sram_bank_num = int(config.get(layout_section, 'FilterSRAMBankNum'))
        self.filter_sram_bank_port = int(config.get(layout_section, 'FilterSRAMBankPort'))
        
        # Anand: ISSUE #2. Patch
        if self.use_user_bandwidth:
            self.bandwidths = [int(x.strip())
                               for x in config.get(section, 'Bandwidth').strip().split(',')]

        if self.df not in self.valid_df_list:
            print("WARNING: Invalid dataflow")

        if config.has_section('network_presets'):  # Read network_presets
            self.topofile = config.get(section, 'TopologyCsvLoc').split('"')[1]

        # Sparsity
        section = 'sparsity'
        if config.get(section, 'SparsitySupport').lower() == 'true':
            self.sparsity_support = True
        else:
            self.sparsity_support = False

        if self.sparsity_support:
            self.sparsity_representation = config.get(section, 'SparseRep')
            # self.sparsity_N = int(config.get(section, 'NonZeroElems'))
            # self.sparsity_M = int(config.get(section, 'BlockSize'))
            if config.get(section, 'OptimizedMapping').lower() == 'true':
                self.sparsity_optimized_mapping = True
            else:
                self.sparsity_optimized_mapping = False

            if self.sparsity_optimized_mapping:
                self.sparsity_block_size = int(config.get(section, 'BlockSize'))
                assert self.sparsity_block_size <= self.array_rows, "ERROR: Invalid block size"

            self.sparsity_rand_seed = int(config.get(section, 'RandomNumberGeneratorSeed'))

        self.valid_conf_flag = True

    #
    def update_from_list(self, conf_list):
        """
        Method to update the parameters through a configuration list.
        """
        if not len(conf_list) > 11:
            print("ERROR: scale_config.update_from_list: "
                  "Incompatible number of elements in the list")

        self.run_name = conf_list[0]
        self.array_rows = int(conf_list[1])
        self.array_cols = int(conf_list[2])
        self.ifmap_sz_kb = int(conf_list[3])
        self.filter_sz_kb = int(conf_list[4])
        self.ofmap_sz_kb = int(conf_list[5])
        self.ifmap_offset = int(conf_list[6])
        self.filter_offset = int(conf_list[7])
        self.ofmap_offset = int(conf_list[8])
        self.df = conf_list[9]
        bw_mode_string = str(conf_list[10])

        assert bw_mode_string in ['CALC', 'USER'], 'Invalid mode of operation'
        if bw_mode_string == "USER":
            assert not len(conf_list) < 12, 'The user bandwidth needs to be provided'
            self.bandwidths = conf_list[11]
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False

        if len(conf_list) == 15:
            self.topofile = conf_list[14]

        self.valid_conf_flag = True

    #
    def write_conf_file(self, conf_file_out):
        """
        Method to generate a configuration file.
        """
        if not self.valid_conf_flag:
            print('ERROR: scale_config.write_conf_file: No valid config loaded')
            return

        config = cp.ConfigParser()

        section = 'general'
        config.add_section(section)
        config.set(section, 'run_name', str(self.run_name))

        section = 'architecture_presets'
        config.add_section(section)
        config.set(section, 'ArrayHeight', str(self.array_rows))
        config.set(section, 'ArrayWidth', str(self.array_cols))

        config.set(section, 'ifmapsramszkB', str(self.ifmap_sz_kb))
        config.set(section, 'filtersramszkB', str(self.filter_sz_kb))
        config.set(section, 'ofmapsramszkB', str(self.ofmap_sz_kb))

        config.set(section, 'IfmapOffset', str(self.ifmap_offset))
        config.set(section, 'FilterOffset', str(self.filter_offset))
        config.set(section, 'OfmapOffset', str(self.ofmap_offset))

        config.set(section, 'Dataflow', str(self.df))
        config.set(section, 'Bandwidth', ','.join([str(x) for x in self.bandwidths]))

        section = 'network_presets'
        config.add_section(section)
        topofile = '"' + self.topofile + '"'
        config.set(section, 'TopologyCsvLoc', str(topofile))

        with open(conf_file_out, 'w') as configfile:
            config.write(configfile)

    #
    def set_arr_dims(self, rows=1, cols=1):
        """
        Method to set the dimensions of the PE array, with default dimensions set to 1x1.
        """
        self.array_rows = rows
        self.array_cols = cols

    #
    def set_dataflow(self, dataflow='os'):
        """
        Method to set the dataflow for the matric multiplication with Output Stationary being the
        default dataflow.
        """
        self.df = dataflow

    #
    def set_buffer_sizes_kb(self, ifmap_size_kb=1, filter_size_kb=1, ofmap_size_kb=1):
        """
        Method to set the IFMAP, Filter and OFMAP SRAM sizes, with the defaults set to 1kB.
        """
        self.ifmap_sz_kb = ifmap_size_kb
        self.filter_sz_kb = filter_size_kb
        self.ofmap_sz_kb = ofmap_size_kb

    #
    def set_topology_file(self, topofile=''):
        """
        Method to set the topology file path.
        """
        self.topofile = topofile
    
    #
    def set_layout_file(self, layoutfile=''):
        self.layoutfile = layoutfile

    #
    def set_offsets(self,
                    ifmap_offset=0,
                    filter_offset=10000000,
                    ofmap_offset=20000000
                    ):
        """
        Method to set the offsets used for IFMAP, Filter and OFMAP addresses, with the defaults set
        to 0, 10M and 20M respectively.
        """
        self.ifmap_offset = ifmap_offset
        self.filter_offset = filter_offset
        self.ifmap_offset = ofmap_offset
        self.valid_conf_flag = True

    #
    def force_valid(self):
        """
        Method to set the 'valid_config_flag' without any checks.
        """
        self.valid_conf_flag = True

    #
    def set_bw_mode_to_calc(self):
        """
        Method to set the 'use_user_bandwidth' to CALC mode.
        """
        self.use_user_bandwidth = False

    #
    def use_user_dram_bandwidth(self):
        """
        Method that returns the value of 'use_user_bandwidth'.
        """
        if not self.valid_conf_flag:
            me = 'scale_config.' + 'use_user_dram_bandwidth()'
            message = 'ERROR: ' + me + ': Configuration is not valid'
            print(message)
            return

        return self.use_user_bandwidth

    #
    def get_conf_as_list(self):
        """
        Method to extract the configuration parameters in the form of a list.
        """
        out_list = []

        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_conf_as_list: Configuration is not valid")
            return

        out_list.append(str(self.run_name))

        out_list.append(str(self.array_rows))
        out_list.append(str(self.array_cols))

        out_list.append(str(self.ifmap_sz_kb))
        out_list.append(str(self.filter_sz_kb))
        out_list.append(str(self.ofmap_sz_kb))

        out_list.append(str(self.ifmap_offset))
        out_list.append(str(self.filter_offset))
        out_list.append(str(self.ofmap_offset))

        out_list.append(str(self.df))
        out_list.append(str(self.topofile))

        return out_list

    #
    def get_run_name(self):
        """
        Method to get the run name used for the simulation.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_run_name() : Config data is not valid")
            return

        return self.run_name

    #
    def get_topology_path(self):
        """
        Method to get the topology file path used for the simulation.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_path() : Config data is not valid")
            return
        return self.topofile

    def get_layout_path(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_layout_path() : Config data is not valid")
            return
        return self.layoutfile

    def get_topology_name(self):
        """
        Method to extract the name of the topology file from the topology path.
        """
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_name() : Config data is not valid")
            return

        name = self.topofile.split('/')[-1].strip()
        name = name.split('.')[0]

        return name

    #
    def get_dataflow(self):
        """
        Method to get the dataflow used for the simulation.
        """
        if self.valid_conf_flag:
            return self.df

    #
    def get_array_dims(self):
        """
        Method to get the dimensions of the PE array.
        """
        if self.valid_conf_flag:
            return self.array_rows, self.array_cols

    #
    def get_mem_sizes(self):
        """
        Method to get the IFMAP, Filter and OFMAP SRAM sizes.
        """
        me = 'scale_config.' + 'get_mem_sizes()'

        if not self.valid_conf_flag:
            message = 'ERROR: ' + me
            message += 'Config is not valid. Not returning any values'
            return

        return self.ifmap_sz_kb, self.filter_sz_kb, self.ofmap_sz_kb

    #
    def get_offsets(self):
        """
        Method to get the offsets used for IFMAP, Filter and OFMAP addresses.
        """
        if self.valid_conf_flag:
            return self.ifmap_offset, self.filter_offset, self.ofmap_offset
    
    def get_ramulator_trace(self):
        """
        Method to check if the run considers ramulator trace numpy files
        """
        if self.valid_conf_flag:
            return self.use_ramulator_trace
    
    def get_req_buf_sz_rd(self):
        """
        Method to set the read request buffer size
        """
        if self.valid_conf_flag:
            return self.req_buf_sz_rd
    
    def get_req_buf_sz_wr(self):
        """
        Method to set the write request buffer size
        """
        if self.valid_conf_flag:
            return self.req_buf_sz_wr
    
    #
    def get_bandwidths_as_string(self):
        """
        Method to get the bandwidths as a string.
        """
        if self.valid_conf_flag:
            return ','.join([str(x) for x in self.bandwidths])

  #
    def get_ifmap_sram_bandwidth(self):
        """
        Method to get the bandwidths as a value.
        """
        if self.valid_conf_flag:
            return self.ifmap_sram_bank_bandwidth

    def get_filter_sram_bandwidth(self):
      """
      Method to get the bandwidths as a value.
      """
      if self.valid_conf_flag:
          return self.filter_sram_bank_bandwidth     

    #
    def get_bandwidths_as_list(self):
        """
        Method to get the bandwidths as a list.
        """
        if self.valid_conf_flag:
            return self.bandwidths

    def get_bandwidths_as_list(self):
        if self.valid_conf_flag:
            return self.bandwidths
        
    def get_num_bank(self):
        if self.valid_conf_flag:
            return self.num_bank
        
    def get_num_port(self):
        if self.valid_conf_flag:
            return self.num_port
        
    def get_min_dram_bandwidth(self):
        """
        Method to get the minimum DRAM bandwidth defined in the configuration.
        """
        if not self.use_user_dram_bandwidth():
            me = 'scale_config.' + 'get_min_dram_bandwidth()'
            message = 'ERROR: ' + me + ': No user bandwidth provided'
            print(message)
        else:
            return min(self.bandwidths)

    # FIX ISSUE #14
    @staticmethod
    def get_default_conf_as_list():
        """
        Method to get the default configuration as a list.
        """
        dummy_obj = scale_config()
        dummy_obj.force_valid()
        out_list = dummy_obj.get_conf_as_list()
        return out_list
