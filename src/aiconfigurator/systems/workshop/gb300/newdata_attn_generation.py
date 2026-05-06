import os
import pandas as pd
import yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def process_extrapolate_generation(sys_old, sys_new, input_csv, output_txt, device_name="NVIDIA VR200"):
    df = pd.read_csv(input_csv)
    
    # 获取原本数据的最小Latency（作为内核启动不可避免的基础耗时）
    min_latency = df['latency'].min()
    print(f"Old baseline minimum latency (kernel launch overhead): {min_latency:.6f} ms")
    
    # 提取旧硬件和新硬件的内存带宽 (Mem BW)
    mem_bw_old = sys_old['gpu']['mem_bw']
    mem_bw_new = sys_new['gpu']['mem_bw']
    
    bw_ratio = mem_bw_old / mem_bw_new
    print(f"Memory bandwidth scaling factor (old / new): {bw_ratio:.4f}")
    
    # 由于全是访存受限 (Memory Bound)，直接按照两代硬件访存带宽反比进行线性缩放
    df['new_latency'] = df['latency'] * bw_ratio
    
    # 极限值保护：缩放后的时间不能低于算子自身的Launch时间瓶颈
    df['new_latency'] = df['new_latency'].clip(lower=min_latency)
    
    df['device'] = device_name
    df['latency'] = df['new_latency']
    
    if 'step' in df.columns:
        df['step'] = df['step'].fillna(0).astype(int)
    
    # 只提取与原始测例集一致的字段进行TXT落盘
    out_columns = [
        'framework', 'version', 'device', 'op_name', 'kernel_source', 
        'mla_dtype', 'kv_cache_dtype', 'num_heads', 'batch_size', 
        'isl', 'tp_size', 'step', 'latency'
    ]
    
    df[out_columns].to_csv(output_txt, index=False)
    print(f"Extrapolated generation latency successfully saved to {output_txt}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "output_generation_mla_perf.csv")
    output_txt = os.path.join(script_dir, "vr200_generation_mla_perf.txt")
    
    sys_gb300_path = os.path.join(script_dir, "..", "gb300.yaml")
    sys_vr200_path = os.path.join(script_dir, "..", "vr200.yaml")
    
    if not os.path.exists(sys_gb300_path) or not os.path.exists(sys_vr200_path):
        print("Error: YAML config files for gb300 or vr200 not found.")
        return
        
    sys_old = load_yaml(sys_gb300_path)
    sys_new = load_yaml(sys_vr200_path)
    print(f"Loaded GB300 (mem_bw: {sys_old['gpu']['mem_bw']}) and VR200 (mem_bw: {sys_new['gpu']['mem_bw']}) hardware configs.")
    
    if os.path.exists(input_csv):
        process_extrapolate_generation(sys_old, sys_new, input_csv, output_txt, device_name="NVIDIA VR200")
    else:
        print(f"Error: Input enriched CSV not found: {input_csv}. Please run data_transfer_attn.py first.")

if __name__ == '__main__':
    main()
