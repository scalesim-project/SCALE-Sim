"""
This file contains the class 'ScalesimReport' containing the utility functions to read the report
data.
"""

import pandas as pd


class ScalesimReport:
    """
    Class which has the utility functions to load report data.
    """
    def __init__(self):
        """
        __init__ method.
        """
        self.compute_df = pd.DataFrame()
        self.bandwidths_df = pd.DataFrame()
        self.details_df = pd.DataFrame()

        self.compute_df_ready = False
        self.bandwidths_df_ready = False
        self.details_df_ready = False

    #
    def load_data(self, data_dir='.', run_name=''):
        """
        Method to load data of compute, bandwidth and detail reports.
        """
        self.load_compute_report_data(data_dir=data_dir, run_name=run_name)
        self.load_bandwidth_report_data(data_dir=data_dir, run_name=run_name)
        self.load_detail_report_data(data_dir=data_dir, run_name=run_name)

    #
    def load_compute_report_data(self, data_dir='.', run_name=''):
        """
        Method to load data of the compute report.
        """
        csv_filename = data_dir + '/' + run_name + '/COMPUTE_REPORT.csv'
        self.compute_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
        self.compute_df_ready = True

    #
    def load_bandwidth_report_data(self, data_dir='.', run_name=''):
        """
        Method to load data of the bandwidth report.
        """
        csv_filename = data_dir + '/' + run_name + '/BANDWIDTH_REPORT.csv'
        self.bandwidths_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
        self.bandwidths_df_ready = True

    #
    def load_detail_report_data(self, data_dir='.', run_name=''):
        """
        Method to load data of the detail report.
        """
        csv_filename = data_dir + '/' + run_name + '/DETAILED_ACCESS_REPORT.csv'
        self.bandwidths_df = pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')
        self.details_df_ready = True

    #
    def get_total_cycles_single_layer(self, layer_id=0):
        """
        Method to get total cycles of a single layer of the workload.
        """
        assert self.compute_df_ready, 'Data not read yet'
        col_name = 'Total Cycles'
        elem = self.compute_df[col_name][layer_id]

        return elem

    #
    def get_compute_cycles_all_layer(self):
        """
        Method to get total cycles of all the layers of the workload.
        """
        assert self.compute_df_ready, 'Data not read yet'
        col_name = 'Total Cycles'
        elems = self.compute_df[col_name].to_list()

        return elems

    #
    def get_sram_ifmap_reads_single_layer(self, layer_id=0):
        """
        Method to get ifmap reads of a single layer of the workload.
        """
        assert self.details_df_ready, 'Data not read yet'
        col_name = 'SRAM IFMAP Reads'
        elem = self.details_df[col_name][layer_id]

        return elem

    #
    def get_sram_ifmap_reads_all_layer(self):
        """
        Method to get ifmap reads of all the layers of the workload.
        """
        assert self.details_df_ready, 'Data not read yet'
        col_name = 'SRAM IFMAP Reads'
        elems = self.details_df[col_name].to_list()

        return elems

if __name__ == '__main__':
    rpt = ScalesimReport()
    rpt.load_data(data_dir='./tutorial1_runs', run_name='scale_sim_tutorial1_64x64_os')

    print(rpt.get_compute_cycles_all_layer())
    print(rpt.get_total_cycles_single_layer(layer_id=1))
