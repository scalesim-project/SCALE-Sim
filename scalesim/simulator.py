"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import os
import pandas as pd
import io

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.layout_utils import layouts as layout
from scalesim.single_layer_sim import single_layer_sim as layer_sim
from scalesim.linear_model.tpu import tpuv4_linear_model, tpuv5e_linear_model, tpuv6e_linear_model


class simulator:
    """
    Class which runs the simulations and manages generated data across various layers
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.conf = cfg()
        self.topo = topo()
        self.layout = layout()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []

        self.params_set_flag = False
        self.all_layer_run_done = False

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   layout_obj=layout(),
                   top_path="./",
                   verbosity=True,
                   save_trace=True,
                   ncu_metrics=''
                   ):
        """
        Method to set the run parameters including inputs and parameters for housekeeping.
        """
        self.conf = config_obj
        self.topo = topo_obj
        self.layout = layout_obj

        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace
        self.ncu_metrics = ncu_metrics
        self.ncu_conv_kernels = []

        if self.ncu_metrics and os.path.exists(self.ncu_metrics):
            self.parse_ncu_metrics(self.ncu_metrics)

        # Calculate inferrable parameters here
        self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True

    def parse_ncu_metrics(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('"ID"'):
                header_idx = i
                break
        if header_idx is None: return

        csv_text = "".join(lines[header_idx:])
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]

        pivot = df.pivot_table(
            index=["ID", "Kernel Name"],
            columns="Metric Name",
            values="Metric Value",
            aggfunc="first",
        ).reset_index()

        conv_keywords = ["gemm", "conv", "fprop", "xmma", "cutlass", "sgemm"]
        exclude_keywords = ["bn_fw", "elementwise", "max_pool", "nchwtonhwc", "nhwctonchw", "offsets"]

        def is_conv_kernel(name):
            name_lower = name.lower()
            if "cudnn::bn" in name_lower: return False
            if any(k in name_lower for k in exclude_keywords): return False
            return any(k in name_lower for k in conv_keywords) or ("cudnn" in name_lower and "ampere_scudnn" in name_lower)

        pivot["is_conv"] = pivot["Kernel Name"].apply(is_conv_kernel)
        
        # Group kernels: Sum all kernels following a conv kernel into that conv kernel's bucket
        # This captures BatchNorm, ReLU, and elementwise overhead belonging to that logical layer.
        layer_buckets = []
        current_layer = None
        
        for _, row in pivot.iterrows():
            if row["is_conv"]:
                if current_layer is not None:
                    layer_buckets.append(current_layer)
                current_layer = {
                    'cycles': row['sm__cycles_active.avg'],
                    'sram_reads': (row.get('l1tex__t_sectors.sum', 0) + row.get('lts__t_sectors.sum', 0)) * 32,
                    'dram_reads': row.get('dram__bytes_read.sum', 0),
                    'dram_writes': row.get('dram__bytes_write.sum', 0)
                }
            elif current_layer is not None:
                # Add overhead of non-conv kernels (ReLU, BN, etc.) to the memory totals
                current_layer['sram_reads'] += (row.get('l1tex__t_sectors.sum', 0) + row.get('lts__t_sectors.sum', 0)) * 32
                current_layer['dram_reads'] += row.get('dram__bytes_read.sum', 0)
                current_layer['dram_writes'] += row.get('dram__bytes_write.sum', 0)
                # Note: We don't sum cycles because SCALE-Sim models math latency, 
                # and these are often overlapping or utility tasks.
        
        if current_layer is not None:
            layer_buckets.append(current_layer)

        # Create a list of dicts for each conv kernel
        self.ncu_conv_kernels = []
        import math
        def safe_int(val):
            if val == "n/a" or val is None: return 0
            try:
                v = float(str(val).replace(',', ''))
                if math.isnan(v) or math.isinf(v): return 0
                return int(v)
            except:
                return 0

        for layer in layer_buckets:
            self.ncu_conv_kernels.append({
                'cycles': safe_int(layer['cycles']),
                'sram_reads': safe_int(layer['sram_reads']),
                'dram_reads': safe_int(layer['dram_reads']),
                'dram_writes': safe_int(layer['dram_writes'])
            })


    #
    def run(self):
        """
        Method to run scalesim simulation for all layers. This method first runs compute and memory
        simulations for each layer and gathers the required stats. Once the simulation runs are
        done, it gathers the stats from single_layer_sim objects and calls generate_report() method
        to create the report files. If save_trace flag is set, then layer wise traces are saved as
        well.
        """
        assert self.params_set_flag, 'Simulator parameters are not set'

        # 1. Create the layer runners for each layer
        for i in range(self.num_layers):
            this_layer_sim = layer_sim()
            this_layer_sim.set_params(layer_id=i,
                                 config_obj=self.conf,
                                 topology_obj=self.topo,
                                 layout_obj=self.layout,
                                 verbose=self.verbose)

            self.single_layer_sim_object_list.append(this_layer_sim)

        if not os.path.isdir(self.top_path):
            os.mkdir(self.top_path)

        report_path = self.top_path + '/' + self.conf.get_run_name()

        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path

        # 2. Run each layer
        # TODO: This is parallelizable
        for single_layer_obj in self.single_layer_sim_object_list:

            if self.verbose:
                layer_id = single_layer_obj.get_layer_id()
                print('\nRunning Layer ' + str(layer_id))

            single_layer_obj.run()

            if self.verbose:
                comp_items = single_layer_obj.get_compute_report_items()
                total_cycles = comp_items[0]
                comp_cycles = comp_items[1]
                stall_cycles = comp_items[2]
                util = comp_items[3]
                mapping_eff = comp_items[4]
                print('Total cycles: ' + str(total_cycles))
                print('Compute cycles: ' + str(comp_cycles))
                print('Stall cycles: ' + str(stall_cycles))
                print('Overall utilization: ' + "{:.2f}".format(util) +'%')
                print('Mapping efficiency: ' + "{:.2f}".format(mapping_eff) +'%')

                avg_bw_items = single_layer_obj.get_bandwidth_report_items()
                if self.conf.sparsity_support is True:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_filter_metadata_sram_bw = avg_bw_items[2]
                    avg_ofmap_sram_bw = avg_bw_items[3]
                    avg_ifmap_dram_bw = avg_bw_items[4]
                    avg_filter_dram_bw = avg_bw_items[5]
                    avg_ofmap_dram_bw = avg_bw_items[6]
                else:
                    avg_ifmap_sram_bw = avg_bw_items[0]
                    avg_filter_sram_bw = avg_bw_items[1]
                    avg_ofmap_sram_bw = avg_bw_items[2]
                    avg_ifmap_dram_bw = avg_bw_items[3]
                    avg_filter_dram_bw = avg_bw_items[4]
                    avg_ofmap_dram_bw = avg_bw_items[5]

                print('Average IFMAP SRAM BW: ' + "{:.3f}".format(avg_ifmap_sram_bw) + \
                      ' words/cycle')
                print('Average Filter SRAM BW: ' + "{:.3f}".format(avg_filter_sram_bw) + \
                      ' words/cycle')
                if self.conf.sparsity_support is True:
                    print('Average Filter Metadata SRAM BW: ' + \
                          "{:.3f}".format(avg_filter_metadata_sram_bw) + ' words/cycle')
                print('Average OFMAP SRAM BW: ' + "{:.3f}".format(avg_ofmap_sram_bw) + \
                      ' words/cycle')
                print('Average IFMAP DRAM BW: ' + "{:.3f}".format(avg_ifmap_dram_bw) + \
                      ' words/cycle')
                print('Average Filter DRAM BW: ' + "{:.3f}".format(avg_filter_dram_bw) + \
                      ' words/cycle')
                print('Average OFMAP DRAM BW: ' + "{:.3f}".format(avg_ofmap_dram_bw) + \
                      ' words/cycle')

            if self.save_trace:
                if self.verbose:
                    print('Saving traces: ', end='')
                single_layer_obj.save_traces(self.top_path)
                if self.verbose:
                    print('Done!')

        self.all_layer_run_done = True

        self.generate_reports()

    #
    def generate_reports(self):
        """
        Method to generate the report files for scalesim run if the runs are already completed. For
        each layer, this method collects the report data from single_layer_sim objects and then
        prints them out into COMPUTE_REPORT.csv, BANDWIDTH_REPORT.csv, DETAILED_ACCESS_REPORT.csv
        and SPARSE_REPORT.csv files.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        compute_report_name = self.top_path + '/COMPUTE_REPORT.csv'
        compute_report = open(compute_report_name, 'w')
        header = ('LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %,'
                  ' Compute Util %,\n')
        compute_report.write(header)
        
        # Create TIME_REPORT.csv for linear model time conversion
        time_report_name = self.top_path + '/TIME_REPORT.csv'
        time_report = open(time_report_name, 'w')
        time_report.write('LayerID, Time (us),\n')

        bandwidth_report_name = self.top_path + '/BANDWIDTH_REPORT.csv'
        bandwidth_report = open(bandwidth_report_name, 'w')
        if self.conf.sparsity_support is True:
            header = ('LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg FILTER Metadata SRAM BW,'
                      ' Avg OFMAP SRAM BW, ')
        else:
            header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '
        header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'
        bandwidth_report.write(header)

        detail_report_name = self.top_path + '/DETAILED_ACCESS_REPORT.csv'
        detail_report = open(detail_report_name, 'w')
        header = 'LayerID, '
        header += 'SRAM IFMAP Start Cycle, SRAM IFMAP Stop Cycle, SRAM IFMAP Reads, '
        header += 'SRAM Filter Start Cycle, SRAM Filter Stop Cycle, SRAM Filter Reads, '
        header += 'SRAM OFMAP Start Cycle, SRAM OFMAP Stop Cycle, SRAM OFMAP Writes, '
        header += 'DRAM IFMAP Start Cycle, DRAM IFMAP Stop Cycle, DRAM IFMAP Reads, '
        header += 'DRAM Filter Start Cycle, DRAM Filter Stop Cycle, DRAM Filter Reads, '
        header += 'DRAM OFMAP Start Cycle, DRAM OFMAP Stop Cycle, DRAM OFMAP Writes,\n'
        detail_report.write(header)

        if self.conf.sparsity_support is True:
            sparse_report_name = self.top_path + '/SPARSE_REPORT.csv'
            sparse_report = open(sparse_report_name, 'w')
            header = 'LayerID, '
            header += 'Sparsity Representation, '
            header += ('Original Filter Storage, New Storage (Filter+Metadata),'
                       ' Filter Metadata Storage, ')
            header += 'Avg FILTER Metadata SRAM BW, '
            header += '\n'
            sparse_report.write(header)

        # Get the number of tensor cores from the config file
        tensor_cores = self.conf.get_tensor_cores()

        for lid in range(len(self.single_layer_sim_object_list)):
            single_layer_obj = self.single_layer_sim_object_list[lid]
            compute_report_items_this_layer = single_layer_obj.get_compute_report_items()
            bandwidth_report_items_this_layer = single_layer_obj.get_bandwidth_report_items()
            detail_report_items_this_layer = single_layer_obj.get_detail_report_items()

            layer_target = self.topo.get_layer_target(lid)

            if layer_target == 'GPU' and len(self.ncu_conv_kernels) > lid:
                ncu_metrics = self.ncu_conv_kernels[lid]
                cycles = ncu_metrics['cycles']
                sram_reads = ncu_metrics['sram_reads']
                dram_reads = ncu_metrics['dram_reads']
                dram_writes = ncu_metrics['dram_writes']

                # Override Compute items: Total, Compute, Stall
                compute_report_items_this_layer[0] = cycles
                compute_report_items_this_layer[1] = cycles
                compute_report_items_this_layer[2] = 0
                
                # Override Detailed Memory Access Counts
                detail_report_items_this_layer[2] = sram_reads
                detail_report_items_this_layer[5] = 0
                detail_report_items_this_layer[8] = 0
                detail_report_items_this_layer[11] = dram_reads
                detail_report_items_this_layer[14] = 0
                detail_report_items_this_layer[17] = dram_writes

                # Override Bandwidth items: simple average
                if self.conf.sparsity_support is True:
                    bandwidth_report_items_this_layer[0] = sram_reads / cycles if cycles > 0 else 0
                    for i in range(1, 4): bandwidth_report_items_this_layer[i] = 0
                    bandwidth_report_items_this_layer[4] = dram_reads / cycles if cycles > 0 else 0
                    bandwidth_report_items_this_layer[5] = 0
                    bandwidth_report_items_this_layer[6] = dram_writes / cycles if cycles > 0 else 0
                else:
                    bandwidth_report_items_this_layer[0] = sram_reads / cycles if cycles > 0 else 0
                    for i in range(1, 3): bandwidth_report_items_this_layer[i] = 0
                    bandwidth_report_items_this_layer[3] = dram_reads / cycles if cycles > 0 else 0
                    bandwidth_report_items_this_layer[4] = 0
                    bandwidth_report_items_this_layer[5] = dram_writes / cycles if cycles > 0 else 0

            elif layer_target == 'GPU' and tensor_cores > 1:
                # Fallback to crude analytical model if NCU metrics not present
                compute_report_items_this_layer[0] /= tensor_cores
                compute_report_items_this_layer[1] /= tensor_cores
                compute_report_items_this_layer[2] /= tensor_cores
                
                for idx in [2, 5, 8, 11, 14, 17]:
                    detail_report_items_this_layer[idx] /= tensor_cores

            log = str(lid) +', '
            log += ', '.join([str(x) for x in compute_report_items_this_layer])
            log += ',\n'
            compute_report.write(log)
            
            # Generate TIME_REPORT entry using linear model
            total_cycles = compute_report_items_this_layer[1]  # Total Cycles (not including prefetch)
            time_linear_model = self.conf.get_time_linear_model()
            
            # Get spatiotemporal dimensions for this layer
            dataflow = self.conf.get_dataflow()
            s_row, s_col, t_time = self.topo.get_spatiotemporal_dims(layer_id=lid, df=dataflow)
            
            
            # Apply the appropriate linear model based on config
            if time_linear_model == 'TPUv4':
                time_us = tpuv4_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv5e':
                time_us = tpuv5e_linear_model(total_cycles, s_row, s_col, t_time)
            elif time_linear_model == 'TPUv6e':
                time_us = tpuv6e_linear_model(total_cycles, s_row, s_col, t_time)
            else:
                # Default: no conversion, just use cycles as time
                time_us = total_cycles
            
            time_log = str(lid) + ', ' + str(time_us) + ',\n'
            time_report.write(time_log)

            log = str(lid) + ', '
            log += ', '.join([str(x) for x in bandwidth_report_items_this_layer])
            log += ',\n'
            bandwidth_report.write(log)

            log = str(lid) + ', '
            log += ', '.join([str(x) for x in detail_report_items_this_layer])
            log += ',\n'
            detail_report.write(log)

            if self.conf.sparsity_support is True:
                sparse_report_items_this_layer = single_layer_obj.get_sparse_report_items()
                log = str(lid) + ', ' + self.conf.sparsity_representation + ', '
                log += ', '.join([str(x) for x in sparse_report_items_this_layer])
                log += ',\n'
                sparse_report.write(log)

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()
        time_report.close()
        if self.conf.sparsity_support is True:
            sparse_report.close()

    #
    def get_total_cycles(self):
        """
        Method which aggregates the total cycles (both compute and stall) across all the layers for
        the given workload.
        """
        assert self.all_layer_run_done, 'Layer runs are not done yet'

        total_cycles = 0
        for layer_obj in self.single_layer_sim_object_list:
            cycles_this_layer = int(layer_obj.get_compute_report_items[0])
            total_cycles += cycles_this_layer

        return total_cycles

