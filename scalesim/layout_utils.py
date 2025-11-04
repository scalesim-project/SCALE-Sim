import math


class layouts(object):

    def __init__(self):
        self.current_layout_name = ""
        self.layout_file_name = ""
        self.layout_arrays = []
        self.spatio_temp_dim_arrays = []
        self.layers_calculated_hyperparams = []
        self.num_layers = 0
        self.layout_load_flag = False
        self.layout_calc_hyper_param_flag = False
        self.layout_calc_spatiotemp_params_flag = False

    # reset layoutlogy parameters
    def reset(self):
        print("All data reset")
        self.current_layout_name = ""
        self.layout_file_name = ""
        self.layout_load_flag = False
        self.layout_arrays = []
        self.num_layers = 0
        self.layout_calc_hyper_param_flag = False
        self.layers_calculated_hyperparams = []

    #
    def load_layer_params_from_list(self, layer_name, elems_list=[]):
        self.layout_file_name = ''
        self.current_layoutname = ''
        self.layer_name = layer_name
        self.append_layout_arrays(layer_name, elems_list)

        self.num_layers += 1
        self.layout_load_flag = True

    #
    def load_arrays(self, layoutfile='', mnk_inputs=False):
        self.load_layout_conv(layoutfile)

    def load_layout_conv(self, layoutfile):
        first = True
        self.layout_file_name = layoutfile.split('/')[-1]
        name_arr = self.layout_file_name.split('.')
        if len(name_arr) > 1:
            self.current_layout_name = self.layout_file_name.split('.')[-2]
        else:
            self.current_layout_name = self.layout_file_name
        f = open(layoutfile, 'r')
        for row in f:
            row = row.strip()
            if first or row == '':
                first = False
            else:
                elems = row.split(',')[:-1]
                # depth-wise convolution
                if 'DP' in elems[0].strip():
                    for dp_layer in range(int(elems[5].strip())):
                        layer_name = elems[0].strip() + "Channel_" + str(dp_layer)
                        elems[5] = str(1)
                        self.append_layout_arrays(layer_name, elems)
                else:
                    layer_name = elems[0].strip()
                    self.append_layout_arrays(layer_name, elems)

        self.num_layers = len(self.layout_arrays)
        self.layout_load_flag = True


    # Write the contents into a csv file
    def write_layout_file(self,
                      path="",
                      filename=""
                      ):
        if path == "":
            print("WARNING: layoutlogy_utils.write_layout_file: No path specified writing to the cwd")
            path = "./" 

        if filename == "":
            print("ERROR: layoutlogy_utils.write_layout_file: No filename provided")
            return

        filename = path + "/" + filename

        if not self.layout_load_flag:
            print("ERROR: layoutlogy_utils.write_layout_file: No data loaded")
            return

        header = [
                    "Layer name",
                    "IFMAP Height Intraline Factor",
                    "IFMAP Width Intraline Factor",
                    "Filter Height Intraline Factor",
                    "Filter Width Intraline Factor",
                    "Channel Intraline Factor",
                    "Num Filter Intraline Factor",
                    "IFMAP Height Intraline Order",
                    "IFMAP Width Intraline Order",
                    "Channel Intraline Order",
                    "IFMAP Height Interline Order",
                    "IFMAP Width Interline Order",
                    "Channel Interline Order",
                    "Num Filter Intraline Order",
                    "Channel Intraline Order",
                    "Filter Height Intraline Order",
                    "Filter Width Intraline Order",
                    "Num Filter Interline Order",
                    "Channel Interline Order",
                    "Filter Height Interline Order",
                    "Filter Width Interline Order"
                ]

        f = open(filename, 'w')
        log = ",".join(header)
        log += ",\n"
        f.write(log)

        for param_arr in self.layout_arrays:
            log = ",".join([str(x) for x in param_arr])
            log += ",\n"
            f.write(log)

        f.close()


    def append_layout_arrays(self, layer_name, elems):
        entry = [layer_name]

        for i in range(1, len(elems)):
            val = int(str(elems[i]).strip())
            entry.append(val)
            if i == 7 and len(elems) < 9:
                entry.append(val)  # Add the same stride in the col direction automatically

        self.layout_arrays.append(entry)


    def get_current_layout_name(self):
        current_layout_name = ""
        if self.layout_load_flag:
            current_layout_name = self.current_layout_name
        else:
            print('Error: get_current_layout_name(): layout file not read')
        return current_layout_name

    def get_num_layers(self):
        if not self.layout_load_flag:
            print("ERROR: layout.get_num_layers: No array loaded")
            return
        return self.num_layers

    #
    def get_layer_ifmap_intraline_factor(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_ifmap_intraline_factor: Invalid layer id")

        return [self.layout_arrays[layer_id][1], self.layout_arrays[layer_id][2], self.layout_arrays[layer_id][5]]

    #
    def get_layer_ifmap_intraline_order(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_ifmap_intraline_order: Invalid layer id")

        return self.layout_arrays[layer_id][7:10]

    #
    def get_layer_ifmap_interline_order(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_ifmap_interline_order: Invalid layer id")

        return self.layout_arrays[layer_id][10:13]

    #
    def get_layer_filter_intraline_factor(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_filter_intraline_factor: Invalid layer id")

        return [self.layout_arrays[layer_id][6], self.layout_arrays[layer_id][5], self.layout_arrays[layer_id][3], self.layout_arrays[layer_id][4]]

    #
    def get_layer_filter_intraline_order(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_filter_intraline_order: Invalid layer id")

        return self.layout_arrays[layer_id][13:17]

    #
    def get_layer_filter_interline_order(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_filter_interline_order: Invalid layer id")

        return self.layout_arrays[layer_id][17:]


    def get_layer_params(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_params: Invalid layer id")
            return
        layer_params = self.layout_arrays[layer_id]
        return layer_params

    def get_layer_id_from_name(self, layer_name=""):
        if (not self.layout_load_flag) or layer_name == "":
            print("ERROR")
            return
        indx = -1
        for i in range(len(self.layout_arrays)):
            if layer_name == self.layout_arrays[i]:
                indx = i
        if indx == -1:
            print("WARNING: Not found")
        return indx

    #
    def get_layer_name(self, layer_id=0):
        if not (self.layout_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: layout.get_layer_name: Invalid layer id")
            return

        name = self.layout_arrays[layer_id][0]
        return str(name)

    #
    def get_layer_names(self):
        if not self.layout_load_flag:
            print("ERROR")
            return
        layer_names = []
        for entry in self.layout_arrays:
            layer_name = str(entry[0])
            layer_names.append(layer_name)
        return layer_names


if __name__ == '__main__':
    import os
    layout = layouts()
    layout.load_arrays(os.path.join(os.getcwd(), "../layouts/GEMM_mnk/vit_s_downscale_KM_KN.csv"))
    for i in range(layout.get_num_layers()):
        ifmap_intraline_factor = layout.get_layer_ifmap_intraline_factor(layer_id=i)
        ifmap_intraline_order = layout.get_layer_ifmap_intraline_order(layer_id=i)
        ifmap_interline_order = layout.get_layer_ifmap_interline_order(layer_id=i)

        filter_intraline_factor = layout.get_layer_filter_intraline_factor(layer_id=i)
        filter_intraline_order = layout.get_layer_filter_intraline_order(layer_id=i)
        filter_interline_order = layout.get_layer_filter_interline_order(layer_id=i)

        print(f"########### Layer {i} ###########")
        print(f"ifmap_intraline_factor={ifmap_intraline_factor}")
        print(f"ifmap_intraline_order={ifmap_intraline_order}")
        print(f"ifmap_interline_order={ifmap_interline_order}")
        print(f"filter_intraline_factor={filter_intraline_factor}")
        print(f"filter_intraline_order={filter_intraline_order}")
        print(f"filter_interline_order={filter_interline_order}")

        print(f"layout.layout_arrays={layout.layout_arrays}")
