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
        # 主要修改：固定为4个工作线程
        self.num_workers = 4

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
    
    # 主要修改：简化为只执行一次仿真，不再接收复杂参数
    def _run_single_sim(self, kv_length, is_gen_text, part, unique_id):
        """执行单次scalesim仿真并返回周期数"""
        topology_path = self.bulid_topologies(kv_len=kv_length, is_gen_text=is_gen_text, part=part, unique_id=unique_id)

        s = scalesim(
            save_disk_space=True, verbose=False, # 减少终端输出
            config=self.config,
            topology=topology_path,
            layout="./layouts/GEMM_mnk/vit_l_KM_KN.csv",
            input_type_gemm=True
        )
        results = s.run_scale(top_path=self.log_path)
        cycles = results[0] if isinstance(results, (tuple, list)) and len(results) >= 1 else results

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

    # 主要修改：线程1的工作函数 - 串行执行文本生成
    def _run_text_generation_thread(self):
        self._append_to_log("=== THREAD 1: Text Generation Started ===")
        total_text_cycles = 0
        parts = ['qkv', 'attn', 'ffn_up', 'ffn_down']

        for i in range(0, self.gen_text_len, self.sample_rate):
            step_end = min(i + self.sample_rate - 1, self.gen_text_len - 1)
            len_average = (i + step_end) // 2
            kv_len = self.kv_cache_init + len_average + 1
            step_cnts = step_end - i + 1

            for part in parts:
                unique_id = f"text_step{i}_{step_end}"
                self._append_to_log(f"Thread 1: Running Text gen steps {i}-{step_end} (part: {part})")
                cycles = self._run_single_sim(kv_len, True, part, unique_id)
                total_text_cycles += cycles * step_cnts * self.num_layer
        
        self._append_to_log(f"=== THREAD 1: Text Generation Completed, Cycles: {total_text_cycles} ===")
        return total_text_cycles

    # 主要修改：线程2,3,4的通用工作函数 - 执行图像生成的某个部分
    def _run_image_generation_thread(self, thread_id, parts_to_run, part_name):
        self._append_to_log(f"=== THREAD {thread_id}: Image Gen ({part_name}) Started ===")
        total_part_cycles = 0
        
        kv_configs = [
            ("full_cache", self.kv_cache_init + self.gen_image_step + self.image_input_len),
            ("without_img", self.kv_cache_without_img + self.gen_image_step + self.image_input_len),
            ("without_text", self.kv_cache_without_text + self.gen_image_step + self.image_input_len)
        ]

        # 1. 计算三种配置下单层的总周期
        single_layer_total_cycles = 0
        for config_name, kv_len in kv_configs:
            for part in parts_to_run:
                unique_id = f"image_{config_name}"
                self._append_to_log(f"Thread {thread_id}: Running Image gen {config_name} (part: {part})")
                cycles = self._run_single_sim(kv_len, False, part, unique_id)
                single_layer_total_cycles += cycles * self.num_layer
        
        # 2. 乘以总步数
        total_part_cycles = single_layer_total_cycles * self.gen_image_step
        
        self._append_to_log(f"=== THREAD {thread_id}: Image Gen ({part_name}) Completed, Cycles: {total_part_cycles} ===")
        return total_part_cycles

    # 主要修改：全新的run_model，用于调度4个粗粒度线程
    def run_model(self):
        """运行完整的Bagel模型仿真（4线程固定并行版）"""
        if os.path.exists(self.result_path):
            os.remove(self.result_path)

        self._append_to_log("======= Bagel Model Simulation Started (4-Thread Fixed Parallel) =======")
        self._append_to_log(f"Config: text_len={self.gen_text_len}, image_steps={self.gen_image_step}, layers={self.num_layer}")

        start_time = datetime.now()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交4个线程任务
            future_text = executor.submit(self._run_text_generation_thread)
            future_img_attn = executor.submit(self._run_image_generation_thread, 2, ['qkv', 'attn'], "Attention")
            future_img_ffn_up = executor.submit(self._run_image_generation_thread, 3, ['ffn_up'], "FFN-Up")
            future_img_ffn_down = executor.submit(self._run_image_generation_thread, 4, ['ffn_down'], "FFN-Down")

            # 获取每个线程的结果
            text_total_cycles = future_text.result()
            img_attn_cycles = future_img_attn.result()
            img_ffn_up_cycles = future_img_ffn_up.result()
            img_ffn_down_cycles = future_img_ffn_down.result()

        # 汇总总周期
        image_total_cycles = img_attn_cycles + img_ffn_up_cycles + img_ffn_down_cycles
        self.total_cycles_all = text_total_cycles + image_total_cycles

        # 记录最终结果
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self._append_to_log("=" * 50)
        self._append_to_log(f"Text Generation Total Cycles: {text_total_cycles}")
        self._append_to_log(f"Image Generation Total Cycles: {image_total_cycles}")
        self._append_to_log(f"  - Attention Part: {img_attn_cycles}")
        self._append_to_log(f"  - FFN-Up Part: {img_ffn_up_cycles}")
        self._append_to_log(f"  - FFN-Down Part: {img_ffn_down_cycles}")
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
    print(f"Results saved to: {bagel.result_path}")


if __name__ == "__main__":
    main()