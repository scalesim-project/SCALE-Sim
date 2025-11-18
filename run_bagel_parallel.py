import os
import json
import csv
from datetime import datetime
from scalesim.scale_sim import scalesim
# 引入多线程相关库
import threading
from concurrent.futures import ThreadPoolExecutor
import time

class Bagel_sim():
    def __init__(self):
        self.kv_cache_init = None
        self.kv_cache_without_img = None
        self.kv_cache_without_text = None
        self.gen_text_len = None
        self.gen_image_step = None
        self.config = "./configs/scale.cfg"
        self.log_path = "./results"
        self.result_path = "./results/bagel/results.log"
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
        # 为日志添加线程锁，防止多线程写入冲突
        self.log_lock = threading.Lock()
        # 定义并行工作线程数
        self.num_workers = 160

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

    def bulid_topologies(self, kv_len=0, is_gen_text=True, part="all", unique_id=''):
        """
        生成Bagel模型单层拓扑结构，并保存到相应的csv文件中
        Args:
        kv_len: KV cache长度
        is_gen_text: True为文本生成，False为图像生成
        part: 'all', 'qkv', 'attn', 'ffn_up', 'ffn_down'，指定生成哪个部分
        unique_id: 唯一标识符，用于生成独立的文件名
        """
        output_dir = os.path.join(os.path.dirname(__file__), "topologies", "bagel")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"layer_{unique_id}_{part}.csv")

        operations = []
        layer_input = self.text_input_len if is_gen_text else self.image_input_len

        # 根据part信息选择性生成操作
        if is_gen_text:
            if part in ['all', 'qkv']:
                operations.extend(
                    [
                        ("Qmap", layer_input, self.dim, self.dim),
                        ("Kmap", layer_input, self.num_head_kv * self.head_dim, self.dim),
                        ("Vmap", layer_input, self.num_head_kv * self.head_dim, self.dim),
                    ]
                )
            if part in ['all', 'attn']:
                for head in range(self.num_head_q):
                    operations.extend([
                        (f"Head_{head}_QKT", layer_input, kv_len, self.head_dim),
                        (f"Head_{head}_SFMXxV", layer_input, self.head_dim, kv_len),
                    ])
                operations.append(("Omap", layer_input, self.dim, self.dim))
            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.upshape, self.dim))
            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.dim, self.upshape))
        else:
            if part in ['all', 'qkv']:
                operations.extend([
                    ("Qmap_text", self.text_attn, self.dim, self.dim),
                    ("Kmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_text", self.text_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Qmap_vae", self.vae_attn, self.dim, self.dim),
                    ("Kmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                    ("Vmap_vae", self.vae_attn, self.num_head_kv * self.head_dim, self.dim),
                ])
            if part in ['all', 'attn']:
                for head in range(self.num_head_q):
                    operations.extend([
                        (f"Head_{head}_QKT", layer_input, kv_len, self.head_dim),
                        (f"Head_{head}_SFMXxV", layer_input, self.head_dim, kv_len),
                    ])
                operations.append(("Omap", layer_input, self.dim, self.dim))
            if part in ['all', 'ffn_up']:
                operations.append(("FFN_up", layer_input, self.upshape, self.dim))
            if part in ['all', 'ffn_down']:
                operations.append(("FFN_down", layer_input, self.dim, self.upshape))
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Layer", "M", "N", "K", ""])
            for op in operations:
                writer.writerow([op[0], op[1], op[2], op[3], ""])
        
        return output_path
    
    def run_sim_once(self, task_params):
        """线程工作函数，执行单次scalesim仿真并返回周期数"""
        kv_length, is_gen_text, part, unique_id, context_msg = task_params

        self._append_to_log(f"Starting: {context_msg} (part: {part})")

        topology_path = self.bulid_topologies(kv_len=kv_length, is_gen_text=is_gen_text, part=part, unique_id=unique_id)

        s = scalesim(
            save_disk_space=True,
            verbose=True,
            config=self.config,
            topology=topology_path,
            layout="./layouts/GEMM_mnk/vit_l_KM_KN.csv",
            input_type_gemm=True
        )
        # 运行仿真，获取周期数
        results = s.run_scale(top_path=self.log_path)
        cycles = results[0] if isinstance(results, (tuple, list)) and len(results) >= 1 else results

        self._append_to_log(f"Finished: {context_msg} (part: {part}), Cycles: {cycles}")

        if os.path.exists(topology_path):
            os.remove(topology_path)
            
        return cycles
    
    def _append_to_log(self, message):
        """追加信息至日志文件，保证线程安全"""
        log_file = self.result_path
        log_dir = os.path.dirname(log_file)

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        with self.log_lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)

        print(log_entry.strip())

    def get_text_gen_tasks(self):
        self._append_to_log(f"=== Preparing Text Generation Tasks (total steps: {self.gen_text_len}) ===")
        tasks = []
        parts = ['qkv', 'attn', 'ffn_up', 'ffn_down']

        for i in range(0, self.gen_text_len, self.sample_rate):
            step_end = min(i + self.sample_rate - 1, self.gen_text_len - 1)
            len_average = (i + step_end) // 2
            kv_len = self.kv_cache_init + len_average + 1
            step_cnts = step_end - i + 1

            # 为每个部分创建任务
            for part in parts:
                unique_id = f"text_step{i}_{step_end}"
                context_msg = f"Text gen steps {i}-{step_end}, kv_len={kv_len}"
                tasks.append(((kv_len, True, part, unique_id, context_msg), step_cnts * self.num_layer))
        return tasks
    
    def get_image_gen_tasks(self):
        self._append_to_log(f"=== Preparing Image Generation Tasks (total steps: {self.gen_image_step}) ===")
        tasks = []
        parts = ['qkv', 'attn', 'ffn_up', 'ffn_down']

        kv_configs = [
            ("full_cache", self.kv_cache_init + self.gen_image_step + self.image_input_len),
            ("without_img", self.kv_cache_without_img + self.gen_image_step + self.image_input_len),
            ("without_text", self.kv_cache_without_text + self.gen_image_step + self.image_input_len)
        ]

        for config_name, kv_len in kv_configs:
            for part in parts:
                unique_id = f"image_{config_name}"
                context_msg = f"Image gen {config_name}, kv_len={kv_len}"

                tasks.append(((kv_len, False, part, unique_id,context_msg), self.num_layer))
        return tasks
    
    def run_model(self):
        """运行完整的Bagel模型仿真"""
        log_file = self.result_path
        if os.path.exists(log_file):
            os.remove(log_file)

        self._append_to_log("======= Bagel Model Simulation Started (Parallel) =======")
        self._append_to_log(f"Config: text_len={self.gen_text_len}, image_steps={self.gen_image_step}, layers={self.num_layer}, workers={self.num_workers}")

        start_time = datetime.now()
        self.total_cycles_all = 0

        self._append_to_log("=== Preparing All Tasks (Text & Image) ===")
        text_tasks_with_multiplier = self.get_text_gen_tasks()
        image_tasks_with_multiplier = self.get_image_gen_tasks()
        
        all_tasks = text_tasks_with_multiplier + image_tasks_with_multiplier

        tasks, multipliers = zip(*all_tasks)
        self._append_to_log("=== Submitting All Tasks to a Single Pool ===")
        total_calculaed_cycles = 0
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = executor.map(self.run_sim_once, tasks)
            
            # 汇总结果
            text_task_count = len(text_tasks_with_multiplier)

            # 计算文本周期
            final_results = zip(results, multipliers)
            text_cycles_list = list(final_results)[:text_task_count]
            text_total_cycles = sum(cycle * multiplier for cycle, multiplier in text_cycles_list)

            # 计算图像周期
            image_cycles_list = list(final_results)[text_task_count:]
            image_single_layer_total_cycles = sum(cycle * multiplier for cycle, multiplier in image_cycles_list)
            image_total_cycles = image_single_layer_total_cycles * self.gen_image_step

            self.total_cycles_all = text_total_cycles + image_total_cycles
            self._append_to_log(f"=== Text Generation Part Completed, total cycles: {text_total_cycles} ===")
            self._append_to_log(f"=== Image Generation Part Completed, total cycles: {image_total_cycles} ===")


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
    
    bagel = Bagel_sim()
    bagel.read_from_json()
    total_cycles = bagel.run_model()
    
    print(f"\nSimulation completed successfully!")
    print(f"Total cycles: {total_cycles}")
    print(f"Results saved to: {os.path.join(bagel.result_path, 'results.log')}")


if __name__ == "__main__":
    main()



