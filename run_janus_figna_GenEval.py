import os
import json
import csv
import math
from datetime import datetime
from scalesim.scale_sim import scalesim

class Bagel_sim():
    def __init__(self):
        self.kv_cache_init = None
        self.kv_cache_without_img = None
        self.kv_cache_without_text = None
        self.gen_text_len = None
        self.gen_image_step = None
        self.config_fp16 = "./configs/scale.cfg"
        self.config_int4 = "./configs/scale.cfg"
        self.log_path = "./results"
        self.result_path = "./results/bagel"
        self.num_layer = 24
        self.text_input_len = 1
        self.image_input_len = 1378
        self.text_attn = 2
        self.vae_attn = 1376
        self.num_head_kv = 16
        self.num_head_q = 16
        self.dim = 2048
        self.head_dim = 128
        self.upshape = 5632
        self.tile = 64
        self.total_cycles_all = 0
        self.sample_rate = 10
        self.ifmapbufsz = 0
        self.filterbufsz = 0
        self.ofmapbufsz = 0
        self.ifmapbw = 0
        self.filterbw = 0
        self.ofmapbw = 0
        self.active_rate = 0.5
        self.array_height = 0
        self.array_width = 0
        self.text_gen_finished_flag = False
        self.text_gen_cycles = 0

    def read_from_json(self, cfg_path=None):
        if cfg_path is None:
            base = os.path.dirname(__file__)
            cfg_path = os.path.join(base, "topologies", "janus", "config_figna_janus_GenEval.json")
        
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            cfg = {}
        except FileNotFoundError:
            print(f"Config file not found: {cfg_path}")
            cfg = {}
        
        self.kv_cache_init = cfg.get("kv_cache_init")
        self.gen_text_len = cfg.get("gen_text_len")
        self.gen_image_step = cfg.get("gen_image_step")
        self.config_fp16 = cfg.get("config_fp16")
        self.config_int4 = cfg.get("config_int4")
        self.log_path = cfg.get("log_path")
        self.result_path = cfg.get("result_path")
        image_len = cfg.get("image_len")
        text_len = cfg.get("text_len")
        self.kv_cache_without_img = self.kv_cache_init - image_len
        self.kv_cache_without_text = self.kv_cache_init - text_len
        self.sample_rate = cfg.get("sample_rate")
        self.sample_rate = cfg.get("ifmapbufsz")
        self.ifmapbufsz = cfg.get("ifmapbufsz")
        self.filterbufsz = cfg.get("filterbufsz")
        self.ofmapbufsz = cfg.get("ofmapbufsz")
        self.ifmapbw = cfg.get("ifmapbw")
        self.filterbw = cfg.get("filterbw")
        self.ofmapbw = cfg.get("ofmapbw")
        self.active_rate = cfg.get("activate_rate")
        self.array_height = cfg.get("array_height")
        self.array_width = cfg.get("array_width")

        text_flag = cfg.get("text_gen_finished_flag", 'False')
        if text_flag == "True":
            self.text_gen_finished_flag = True
        else:
            self.text_gen_finished_flag = False
        
        self.text_gen_cycles = cfg.get("text_gen_cycles", 0)
        self.tile = cfg.get("tile")

    def build_topologies(self, kv_len=0, is_gen_text=True, part='all'):
        """
        生成Bagel模型单层拓扑结构，并保存到相应的csv文件中

        Args:
        kv_len: KV cache长度，影响attention矩阵大小
        is_gen_text: True为文本生成，False为图像生成
        part: 指定生成哪个部分
        """
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "topologies", "bagel")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "layer.csv")
        
        # 生成矩阵操作列表
        operations = []
        layer_input = self.text_input_len if is_gen_text else self.image_input_len

        # 如果文件存在，先删除以确保全新生成
        if os.path.exists(output_path):
            os.remove(output_path)

        # Q, K, V 映射
        if is_gen_text:
            if part in ['all', 'qkv']:
                operations.extend([
                    ("QKVmap", layer_input, self.tile, self.dim)
                ])
        
            # Multi-head attention操作
            if part in ['all', 'attn_qk']:
                operations.extend([
                    (f"Head_QKT", layer_input, self.tile, self.head_dim)
                ])
            if part in ['all', 'attn_sfmxv']:
                operations.extend([
                    (f"Head_SFMXxV", layer_input, self.tile, kv_len)
                ])

            if part in ['all', 'omap']:
                # 输出映射
                operations.append(("Omap", layer_input, self.tile, self.dim))
        
            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.tile, self.dim))

            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.tile, self.upshape))
        else:
            if part in ['all', 'qkv']:
                operations.extend([
                    ("Qmap_text", self.text_attn, self.dim, self.dim),
                    ("Kmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                ])

                """operations.extend([
                    ("Qmap_vae", self.vae_attn, self.dim, self.dim),
                    ("Kmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                ])"""

            # Multi-head attention操作
            if part in ['all', 'attn']:
                operations.extend([
                    (f"Head_QKT", layer_input, kv_len, self.head_dim),
                    (f"Head_SFMXxV", layer_input, self.head_dim, kv_len),
                ])

            if part in ['all', 'omap']:
                # 输出映射
                operations.append(("Omap", layer_input, self.dim, self.dim))
        

            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.upshape, self.dim))

            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.dim, self.upshape))
        
        # 写入CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Layer", "M", "N", "K", ""])  # 表头
            for op in operations:
                writer.writerow([op[0], op[1], op[2], op[3], ""])
        
        print(f"Topology saved to: {output_path}")
        return output_path

    def run_sim_once(self, kv_length=0, is_gen_text=True, part='all', config="./configs/scale.cfg"):
        topology_path = self.build_topologies(kv_len=kv_length, is_gen_text=is_gen_text, part=part)

        s = scalesim(
            save_disk_space=True,
            verbose=True,
            config=config,
            topology=topology_path,
            layout="./layouts/GEMM_mnk/vit_l_KM_KN.csv",
            input_type_gemm=True
        )

        results = s.run_scale(top_path=self.log_path)

        # 解包结果，获取总周期数
        if isinstance(results, (tuple, list)) and len(results) >= 1:
            return results[0]  # 返回总周期数
        else:
            return results
    
    def _append_to_log(self, message):
        """追加信息到日志文件"""
        # 确保结果目录存在
        log_file = self.result_path
        log_dir = os.path.dirname(log_file)

        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # 追加到日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        print(log_entry.strip())  
        
    def run_gen_text(self):
        self._append_to_log(f"=========== Text Generation Started (total steps: {self.gen_text_len}) ===========")
        text_start_cycles = self.total_cycles_all

        self._append_to_log(f"Text generation - Start qkv mapping")
        input_tile_cycles = self.run_sim_once(kv_length=0, is_gen_text=True, part='qkv', config=self.config_int4)
        input_cycles = input_tile_cycles * (self.dim + self.num_head_kv * self.head_dim * 2) / self.tile
        self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")

        self._append_to_log(f"Text generation - Start output mapping")
        output_tile_cycles = self.run_sim_once(kv_length=0, is_gen_text=True, part='omap', config=self.config_int4)
        output_cycles = output_tile_cycles * self.dim / self.tile
        self._append_to_log(f"Text generation - End output mapping, total cycles:{output_cycles}")

        self._append_to_log(f"Text generation - Start FFN up")
        ffn_up_tile_cycles = self.run_sim_once(kv_length=0, is_gen_text=True, part='ffn_up', config=self.config_int4)
        ffn_up_cycles = ffn_up_tile_cycles * self.upshape / self.tile
        self._append_to_log(f"Text generation - End FFN up, total cycles:{ffn_up_cycles}")

        self._append_to_log(f"Text generation - Start FFN down")
        ffn_down_tile_cycles = self.run_sim_once(kv_length=0, is_gen_text=True, part='ffn_down', config=self.config_int4)
        ffn_down_cycles = ffn_down_tile_cycles * self.dim / self.tile
        self._append_to_log(f"Text generation - End FFN down, total cycles:{ffn_down_cycles}")

        for step in range(0, self.gen_text_len, self.sample_rate):
            if step + self.sample_rate > self.gen_text_len:
                step_end = self.gen_text_len - 1
            else:
                step_end = step + self.sample_rate - 1

            len_average = (step + step_end) // 2   
            kv_len = self.kv_cache_init + step + len_average + 1

            self._append_to_log(f"Text generation - Step {step} to Step {step_end} started (kv_len in averagr: {kv_len})")

            results_this_iter = 0
            results_this_iter += input_cycles
            attn_qk_tile_single_cycle = self.run_sim_once(kv_length=kv_len, is_gen_text=True, part='attn_qk', config=self.config_fp16)
            attn_qk_single_cycle = attn_qk_tile_single_cycle * kv_len / self.tile
            attn_sfmxv_tile_single_cycle = self.run_sim_once(kv_length=kv_len, is_gen_text=True, part='attn_sfmxv', config=self.config_fp16)
            attn_sfmxv_single_cycle = attn_sfmxv_tile_single_cycle * self.head_dim / self.tile
            attn_single_cycle = attn_qk_single_cycle + attn_sfmxv_single_cycle
            attn_cycles = attn_single_cycle * self.num_head_q
            results_this_iter += attn_cycles
            results_this_iter += output_cycles
            results_this_iter += ffn_up_cycles
            results_this_iter += ffn_down_cycles

            step_cycles = results_this_iter * self.num_layer

            step_cnts = step_end - step + 1
            step_cycles_total = step_cycles * step_cnts
            self.total_cycles_all += step_cycles_total


        text_total_cycles = self.total_cycles_all - text_start_cycles
        self._append_to_log(f"=========== Text Generation Completed, total cycles: {text_total_cycles}, total seconds: {text_total_cycles/500000000} ===========")

    def run_sim_once_comp(self, kv_len=0, is_gen_text=False, part='all'):
        cycle_result = 0
        layer_input = self.text_input_len if is_gen_text else kv_len
        if part in ['all', 'prefetch']:
            size_a = self.text_attn * self.dim
            size_b = self.dim * self.dim
            if size_a > self.active_rate * self.ifmapbufsz *1024:
                prefetch_cycles_a = self.active_rate * self.ifmapbufsz * 1024 / self.ifmapbw
            else:
                prefetch_cycles_a = size_a / self.ifmapbw

            if size_b > self.active_rate * self.filterbufsz *1024:
                prefetch_cycles_b = self.active_rate * self.filterbufsz * 1024 / self.filterbw
            else:
                prefetch_cycles_b = size_b / self.filterbw
            
            prefetch_cycles = max(prefetch_cycles_a, prefetch_cycles_b)
            cycle_result += prefetch_cycles

        if part in ['all', 'qkv']:
            Qmap_row_fold = math.ceil(layer_input/self.array_height)
            Qmap_col_fold = math.ceil(self.dim/self.array_width)
            Qmap_cycle_each_fold = self.dim + self.array_height + self.array_width - 2
            Qmap_cycles = Qmap_cycle_each_fold * Qmap_col_fold * Qmap_row_fold

            cycle_result += Qmap_cycles

            KVmap_row_fold = math.ceil(layer_input/self.array_height)
            KVmap_col_fold = math.ceil(self.num_head_kv * self.head_dim/self.array_width)
            KVmap_cycle_each_fold = self.dim + self.array_height + self.array_width - 2
            KVmap_cycles = KVmap_cycle_each_fold * KVmap_col_fold * KVmap_row_fold

            cycle_result += 2 * KVmap_cycles
            

        # Multi-head attention操作
        if part in ['all', 'attn']:
            QKT_row_fold = math.ceil(layer_input/self.array_height)
            QKT_col_fold = math.ceil(kv_len/self.array_width)
            QKT_cycle_each_fold = self.head_dim + self.array_height + self.array_width - 2
            QKT_cycles = QKT_cycle_each_fold * QKT_col_fold * QKT_row_fold

            cycle_result += QKT_cycles

            SFMXxV_row_fold = math.ceil(layer_input/self.array_height)
            SFMXxV_col_fold = math.ceil(self.head_dim/self.array_width)
            SFMXxV_cycle_each_fold = kv_len + self.array_height + self.array_width - 2
            SFMXxV_cycles = SFMXxV_cycle_each_fold * SFMXxV_col_fold * SFMXxV_row_fold

            cycle_result += SFMXxV_cycles

        if part in ['all', 'omap']:
            # 输出映射
            Omap_row_fold = math.ceil(layer_input/self.array_height)
            Omap_col_fold = math.ceil(self.dim/self.array_width)
            Omap_cycle_each_fold = self.dim + self.array_height + self.array_width - 2
            Omap_cycles = Omap_cycle_each_fold * Omap_col_fold * Omap_row_fold

            cycle_result += Omap_cycles
    

        if part in ['all', 'ffn_up']:
            FFN_up_row_fold = math.ceil(layer_input/self.array_height)
            FFN_up_col_fold = math.ceil(self.upshape/self.array_width)
            FFN_up_cycle_each_fold = self.dim + self.array_height + self.array_width - 2
            FFN_up_cycles = FFN_up_cycle_each_fold * FFN_up_col_fold * FFN_up_row_fold

            cycle_result += FFN_up_cycles * 2

        if part in ['all', 'ffn_down']:
            FFN_down_row_fold = math.ceil(layer_input/self.array_height)
            FFN_down_col_fold = math.ceil(self.upshape/self.array_width)
            FFN_down_cycle_each_fold = self.dim + self.array_height + self.array_width - 2
            FFN_down_cycles = FFN_down_cycle_each_fold * FFN_down_col_fold * FFN_down_row_fold

            cycle_result += FFN_down_cycles

        if part in ['all', 'drain']:

            drain_cycles = self.active_rate * self.ofmapbufsz * 1024 / self.filterbw
            cycle_result += drain_cycles

        return cycle_result


    def run_gen_image(self):
        self._append_to_log(f"=== Image Generation Started (total steps: {self.gen_image_step}) ===")
        image_start_cycles = self.total_cycles_all

       # 三个不同KV配置的图像生成
        kv_full = self.kv_cache_init + self.gen_text_len + self.image_input_len
        kv_without_text = self.image_input_len
        kv_configs = [
            ("full_cache", kv_full),
            ("without_text", kv_without_text)
        ]

        total_image_cycles = 0
        self._append_to_log(f"Image generation - Start prefetching")
        prefetch_cycles = self.run_sim_once_comp(kv_len=kv_full, is_gen_text=False, part='prefetch')
        self._append_to_log(f"Image generation - End prefetching, total cycles:{prefetch_cycles}")

        self._append_to_log(f"Image generation - Start draining")
        drain_cycles = self.run_sim_once_comp(kv_len=0, is_gen_text=False, part='drain')
        self._append_to_log(f"Image generation - End draining, total cycles:{drain_cycles}")

        total_image_cycles += prefetch_cycles

        for config_name, kv_len in kv_configs:
            self._append_to_log(f"Image generation - {config_name} config started (kv_len: {kv_len})")
            config_cycles_iter = 0

            self._append_to_log(f"Image generation - Start qkv mapping")
            input_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='qkv')
            input_cycles = input_cycles
            self._append_to_log(f"Text generation - End qkv mapping, total cycles:{input_cycles}")
            config_cycles_iter += input_cycles

            self._append_to_log(f"Image generation - Start attn")
            attn_single_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='attn')
            attn_cycles = attn_single_cycles * self.num_head_q
            self._append_to_log(f"Image generation - End attn, total cycles:{attn_cycles}")
            config_cycles_iter += attn_cycles

            self._append_to_log(f"Image generation - Start output mapping")
            output_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='omap')
            self._append_to_log(f"Image generation - End output mapping, total cycles:{output_cycles}")
            config_cycles_iter += output_cycles

            self._append_to_log(f"Image generation - Start FFN up")
            ffn_up_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='ffn_up')
            self._append_to_log(f"Image generation - End FFN up, total cycles:{ffn_up_cycles}")
            config_cycles_iter += ffn_up_cycles

            self._append_to_log(f"Image generation - Start FFN down")
            ffn_down_cycles = self.run_sim_once_comp(kv_len=kv_len, is_gen_text=False, part='ffn_down')
            self._append_to_log(f"Image generation - End FFN down, total cycles:{ffn_down_cycles}")
            config_cycles_iter += ffn_down_cycles

            config_cycles = config_cycles_iter * self.num_layer
            total_image_cycles += config_cycles
            
            self._append_to_log(f"Image generation - {config_name} config completed, cycles: {config_cycles}")

        total_image_cycles += drain_cycles

        # 乘以生成步数
        final_image_cycles = total_image_cycles * self.gen_image_step
        self.total_cycles_all += final_image_cycles
        
        image_total_cycles = self.total_cycles_all - image_start_cycles
        self._append_to_log(f"=== Image Generation Completed, total cycles: {image_total_cycles}, total seconds: {image_total_cycles/500000000} ===")


    def run_model(self):
        """运行完整的Bagel模型仿真"""
        # 清空之前的日志文件
        log_file = os.path.join(self.result_path, "results.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # 记录仿真开始
        self._append_to_log("======= Bagel Model Simulation Started =======")
        self._append_to_log(f"Configuration: text_len={self.gen_text_len}, image_steps={self.gen_image_step}, layers={self.num_layer}")
        
        start_time = datetime.now()
        
        # 重置总周期数
        self.total_cycles_all = 0
        
        # 运行文本生成
        self.run_gen_text()
        
        # 运行图像生成
        self.run_gen_image()
        
        # 记录最终结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self._append_to_log("=" * 50)
        self._append_to_log(f"FINAL RESULT - Total Cycles: {self.total_cycles_all}, Total seconds: {self.total_cycles_all/500000000}")
        self._append_to_log(f"Simulation Duration: {duration:.2f} seconds")
        self._append_to_log("======= Bagel Model Simulation Completed =======")
        
        return self.total_cycles_all


def main():
    """主函数"""
    print("Bagel Model Simulation Starting...")
    
    # 实例化Bagel仿真类
    bagel = Bagel_sim()
    
    # 读取JSON配置
    bagel.read_from_json()
    
    # 运行完整模型仿真
    total_cycles = bagel.run_model()
    
    print(f"\nSimulation completed successfully!")
    print(f"Total cycles: {total_cycles}")
    print(f"Results saved to: {bagel.result_path}/results.log")


if __name__ == "__main__":
    main()