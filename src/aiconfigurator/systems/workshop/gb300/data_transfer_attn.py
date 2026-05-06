import os
import pandas as pd
import yaml
import numpy as np

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_kv_mem(dtype):
    if dtype in ['float16', 'fp16']:
        return 2
    if dtype in ['fp8', 'int8']:
        return 1
    raise ValueError(f"Unknown kv dtype {dtype}")

def get_fmha_compute(dtype):
    if dtype in ['float16', 'fp16']:
        return 1
    if dtype in ['fp8', 'fp8_block']:
        return 2
    raise ValueError(f"Unknown fmha dtype {dtype}")

def process_context_mla(sys_spec, input_file, output_file):
    df = pd.read_csv(input_file)
    b = df['batch_size']
    s = df['isl']
    prefix = 0  # 原始数据中无prefix
    full_s = s + prefix
    num_heads = df['num_heads']
    
    kv_mem = df['kv_cache_dtype'].apply(get_kv_mem)
    fmha_compute = df['mla_dtype'].apply(get_fmha_compute)
    # print("fmha_compute", fmha_compute.unique())
    
    ops = b * num_heads * 320 * (full_s * full_s - prefix * prefix)
    mem_bytes = b * num_heads * (kv_mem * full_s * 320 + 2 * s * 320)
    
    float16_tc_flops = sys_spec['gpu']['float16_tc_flops']
    mem_bw = sys_spec['gpu']['mem_bw']
    
    sol_math = ops / float16_tc_flops * 1000.0 / fmha_compute
    sol_mem = mem_bytes / mem_bw * 1000.0
    
    tc_flops_effective = float16_tc_flops * fmha_compute
    intensity_op = np.where(mem_bytes > 0, ops / mem_bytes, 0)
    intensity_hw = tc_flops_effective / mem_bw
    rel_ratio = np.where(intensity_hw > 0, intensity_op / intensity_hw, 0)
    
    latency = df['latency']
    mfu = np.where(latency > 0, sol_math / latency, 0)
    mbu = np.where(latency > 0, sol_mem / latency, 0)
    
    df['ops'] = ops
    df['mem_bytes'] = mem_bytes
    df['intensity_op'] = intensity_op
    df['intensity_hw'] = intensity_hw
    df['rel_ratio'] = rel_ratio
    df['sol_math'] = sol_math
    df['sol_mem'] = sol_mem
    df['mfu'] = mfu
    df['mbu'] = mbu
    
    df.to_csv(output_file, index=False)
    print(f"Saved context data to {output_file}")


def process_generation_mla(sys_spec, input_file, output_file):
    df = pd.read_csv(input_file)
    b = df['batch_size']
    s = df['isl']
    step = df.get('step', pd.Series(np.zeros(len(df))))
    total_s = s + step
    
    num_heads = df['num_heads']
    kv_mem = df['kv_cache_dtype'].apply(get_kv_mem)
    fmha_compute = np.where(df['kv_cache_dtype'] == 'fp8', 2.0, 1.0)
    # fmha_compute = df['mla_dtype'].apply(get_fmha_compute)
    # print("fmha_compute", fmha_compute.unique())
    
    ops = 2 * b * num_heads * 1088 * total_s
    mem_bytes = b * (num_heads * 1088 * 2 + (total_s - 1) * 576 * kv_mem)
    
    float16_tc_flops = sys_spec['gpu']['float16_tc_flops']
    mem_bw = sys_spec['gpu']['mem_bw']
    
    sol_math = ops / float16_tc_flops * 1000.0 / fmha_compute
    sol_mem = mem_bytes / mem_bw * 1000.0
    
    tc_flops_effective = float16_tc_flops * fmha_compute
    intensity_op = np.where(mem_bytes > 0, ops / mem_bytes, 0)
    intensity_hw = tc_flops_effective / mem_bw
    rel_ratio = np.where(intensity_hw > 0, intensity_op / intensity_hw, 0)
    
    latency = df['latency']
    mfu = np.where(latency > 0, sol_math / latency, 0)
    mbu = np.where(latency > 0, sol_mem / latency, 0)
    
    df['ops'] = ops
    df['mem_bytes'] = mem_bytes
    df['intensity_op'] = intensity_op
    df['intensity_hw'] = intensity_hw
    df['rel_ratio'] = rel_ratio
    df['sol_math'] = sol_math
    df['sol_mem'] = sol_mem
    df['mfu'] = mfu
    df['mbu'] = mbu
    
    df.to_csv(output_file, index=False)
    print(f"Saved generation data to {output_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, "..", "gb300.yaml")
    sys_spec = load_yaml(yaml_path)
    
    data_dir = os.path.join(script_dir, "..", "data", "gb200", "trtllm", "1.2.0rc6")
    context_file = os.path.join(data_dir, "context_mla_perf.txt")
    gen_file = os.path.join(data_dir, "generation_mla_perf.txt")
    
    out_context_file = os.path.join(script_dir, "output_context_mla_perf.csv")
    out_gen_file = os.path.join(script_dir, "output_generation_mla_perf.csv")
    
    if os.path.exists(context_file):
        process_context_mla(sys_spec, context_file, out_context_file)
    else:
        print(f"Context file not found: {context_file}")
        
    if os.path.exists(gen_file):
        process_generation_mla(sys_spec, gen_file, out_gen_file)
    else:
        print(f"Generation file not found: {gen_file}")

if __name__ == '__main__':
    main()
