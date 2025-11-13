import os
import json
import csv
from datetime import datetime
from scalesim.scale_sim import scalesim

class Bagel_sim():
    def __init__(self):
        self.kv_cache_init = None
        self.kv_cache_without_img = None
        self.kv_cache_without_text = None
        self.gen_text_len = None
        self.gen_image_step = None
        self.config = "./configs/scale.cfg"
        self.log_path = "./results"
        self.result_path = "./results/bagel"
        self.num_layer = 28
        self.text_input_len = 1
        self.image_input_len = 1378
        self.text_attn = 2
        self.vae_attn = 1376
        self.num_head_kv = 4
        self.num_head_q = 28
        self.dim = 3584
        self.head_dim = 128
        self.upshape = 18944
        self.total_cycles_all = 0
        self.sample_rate = 10

    def read_from_json(self, cfg_path=None):
        if cfg_path is None:
            base = os.path.dirname(__file__)
            cfg_path = os.path.join(base, "topologies", "bagel", "config.json")
        
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
        self.config = cfg.get("config")
        self.log_path = cfg.get("log_path")
        self.result_path = cfg.get("result_path")
        image_len = cfg.get("image_len")
        text_len = cfg.get("text_len")
        self.kv_cache_without_img = self.kv_cache_init - image_len
        self.kv_cache_without_text = self.kv_cache_init - text_len
        self.sample_rate = cfg.get("sample_rate")

    def build_topologies(self, kv_len=0, is_gen_text=True):
        """
        生成Bagel模型单层拓扑结构，并保存到相应的csv文件中

        Args:
        kv_len: KV cache长度，影响attention矩阵大小
        is_gen_text: True为文本生成，False为图像生成
        """
        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(__file__), "topologies", "bagel")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "layer.csv")
        
        # 生成矩阵操作列表
        operations = []

        # 如果文件存在，先删除以确保全新生成
        if os.path.exists(output_path):
            os.remove(output_path)

        # Q, K, V 映射
        if is_gen_text:
            layer_input = self.text_input_len
            operations.extend([
                ("Qmap", layer_input, self.dim, self.dim),
                ("Kmap", layer_input, self.num_head_kv * self.head_dim, self.dim),
                ("Vmap", layer_input, self.num_head_kv * self.head_dim, self.dim),
            ])
        
            # Multi-head attention操作
            for head in range(self.num_head_q):
                kv_head = head // (self.num_head_q // self.num_head_kv)  # KV head索引
                operations.extend([
                    (f"Head_{head}_QKT", layer_input, kv_len, self.head_dim),
                    (f"Head_{head}_SFMXxV", layer_input, self.head_dim, kv_len),
                ])
            
            # 输出映射
            operations.append(("Omap", layer_input, self.dim, self.dim))
        

            operations.extend([
                ("FFN_up", layer_input, self.upshape, self.dim),
                ("FFN_down", layer_input, self.dim, self.upshape),
            ])
        else:
            layer_input = self.image_input_len
            operations.extend([
                ("Qmap_text", self.text_attn, self.dim, self.dim),
                ("Kmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                ("Vmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
            ])

            operations.extend([
                ("Qmap_vae", self.vae_attn, self.dim, self.dim),
                ("Kmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                ("Vmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
            ])

            # Multi-head attention操作
            for head in range(self.num_head_q):
                kv_head = head // (self.num_head_q // self.num_head_kv)  # KV head索引
                operations.extend([
                    (f"Head_{head}_QKT", layer_input, kv_len, self.head_dim),
                    (f"Head_{head}_SFMXxV", layer_input, self.head_dim, kv_len),
                ])
            
            # 输出映射
            operations.append(("Omap", layer_input, self.dim, self.dim))
        

            operations.extend([
                ("FFN_up", layer_input, self.upshape, self.dim),
                ("FFN_down", layer_input, self.dim, self.upshape),
            ])
        
        # 写入CSV文件
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Layer", "M", "N", "K", ""])  # 表头
            for op in operations:
                writer.writerow([op[0], op[1], op[2], op[3], ""])
        
        print(f"Topology saved to: {output_path}")
        return output_path

    def run_sim_once(self, kv_length=0, is_gen_text=True):
        topology_path = self.build_topologies(kv_len=kv_length, is_gen_text=is_gen_text)

        s = scalesim(
            save_disk_space=True,
            verbose=True,
            config=self.config,
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
        
    def run_text_per_iter(self, step=0):
        if step + self.sample_rate > self.gen_text_len:
            step_end = self.gen_text_len - 1
        else:
            step_end = step + self.sample_rate - 1

        len_average = (step + step_end) // 2   
        kv_len = self.kv_cache_init + step + len_average + 1

        # 记录开始信息
        self._append_to_log(f"Text generation - Step {step} to Step {step_end} started (kv_len in averagr: {kv_len})")

        results = self.run_sim_once(kv_length=kv_len, is_gen_text=True)
        step_cycles = results * self.num_layer

        step_cnts = step_end - step + 1
        step_cycles_total = step_cycles * step_cnts
        
        # 累加到总周期数
        self.total_cycles_all += step_cycles_total

        # 记录完成信息
        self._append_to_log(f"Text generation - Step {step} to Step {step_end} completed, cycles: {step_cycles} * {step_cnts} = {step_cycles_total}")

    def run_gen_text(self):
        self._append_to_log(f"=== Text Generation Started (total steps: {self.gen_text_len}) ===")
        text_start_cycles = self.total_cycles_all

        for i in range(0, self.gen_text_len, self.sample_rate):
            self.run_text_per_iter(step=i)

        text_total_cycles = self.total_cycles_all - text_start_cycles
        self._append_to_log(f"=== Text Generation Completed, total cycles: {text_total_cycles} ===")

    def run_gen_image(self):
        self._append_to_log(f"=== Image Generation Started (total steps: {self.gen_image_step}) ===")
        image_start_cycles = self.total_cycles_all

       # 三个不同KV配置的图像生成
        kv_configs = [
            ("full_cache", self.kv_cache_init + self.gen_image_step + self.image_input_len),
            ("without_img", self.kv_cache_without_img + self.gen_image_step + self.image_input_len),
            ("without_text", self.kv_cache_without_text + self.gen_image_step + self.image_input_len)
        ]

        total_image_cycles = 0

        for config_name, kv_len in kv_configs:
            self._append_to_log(f"Image generation - {config_name} config started (kv_len: {kv_len})")
            
            results = self.run_sim_once(kv_length=kv_len, is_gen_text=False)
            config_cycles = results
            total_image_cycles += config_cycles
            
            self._append_to_log(f"Image generation - {config_name} config completed, cycles: {config_cycles}")
        
        # 乘以生成步数
        final_image_cycles = total_image_cycles * self.gen_image_step
        self.total_cycles_all += final_image_cycles
        
        image_total_cycles = self.total_cycles_all - image_start_cycles
        self._append_to_log(f"=== Image Generation Completed, total cycles: {image_total_cycles} ===")


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
        self._append_to_log(f"FINAL RESULT - Total Cycles: {self.total_cycles_all}")
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