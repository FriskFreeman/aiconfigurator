import os
import pandas as pd
import yaml
import numpy as np

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_fmha_compute(dtype):
    if dtype in ['float16', 'fp16']:
        return 1
    if dtype in ['fp8', 'fp8_block']:
        return 2
    raise ValueError(f"Unknown fmha dtype {dtype}")

def process_extrapolate_context(sys_new, input_csv, output_txt, device_name="NVIDIA VR200"):
    df = pd.read_csv(input_csv)
    
    # 修改1：选取原始旧表中的最小latency作为算子下界（即最基本的启动时间）
    min_latency = df['latency'].min()
    print(f"Old baseline minimum latency (kernel launch overhead): {min_latency:.6f} ms")
    
    mem_bw_new = sys_new['gpu']['mem_bw']
    float16_tc_flops_new = sys_new['gpu']['float16_tc_flops']
    
    # 考虑计算精度的修正项（mla_dtype 对应 fmha_compute）
    df['fmha_compute'] = df['mla_dtype'].apply(get_fmha_compute)
    df['tc_flops_effective_new'] = float16_tc_flops_new * df['fmha_compute']
    
    # 修改2：计算新机器下的各项理论极限（基于原始算子 Ops 和 Mem）
    df['new_sol_math'] = df['ops'] / df['tc_flops_effective_new'] * 1000.0
    df['new_sol_mem'] = df['mem_bytes'] / mem_bw_new * 1000.0
    
    # 修改3：新硬件强制强度比以及算子在目标设备上的理论映射比(new rel_ratio)
    df['new_intensity_hw'] = df['tc_flops_effective_new'] / mem_bw_new
    df['new_rel_ratio'] = np.where(df['new_intensity_hw'] > 0, df['intensity_op'] / df['new_intensity_hw'], 0)
    
    new_latencies = []
    
    # 分组：固定模型结构量即同kv类型、头数、批次，通过序列长度长短变化勾勒出的曲线进行平滑插值
    groups = df.groupby(['kv_cache_dtype', 'num_heads', 'batch_size'])
    
    for idx, row in df.iterrows():
        kv = row['kv_cache_dtype']
        nh = row['num_heads']
        bs = row['batch_size']
        
        # 定位到对应的数据池
        group = groups.get_group((kv, nh, bs)).copy()
        # 必须依赖旧的 rel_ratio 排列构建严格单调或递增的x轴映射坐标
        group = group.sort_values('rel_ratio')
        
        old_rr = group['rel_ratio'].values
        old_mbu = group['mbu'].values
        old_mfu = group['mfu'].values
        
        new_rr = row['new_rel_ratio']
        
        # 插值策略：如果新平台偏访存(<=1)，取MBU体系规律；如果偏计算(>1)，按MFU推导
        if new_rr <= 1.0:
            if len(old_rr) > 1:
                interp_util = np.interp(new_rr, old_rr, old_mbu)
            else:
                interp_util = old_mbu[0] # 数据点单一直接退化
                
            lat = row['new_sol_mem'] / interp_util if interp_util > 0 else row['new_sol_mem']
        else:
            if len(old_rr) > 1:
                interp_util = np.interp(new_rr, old_rr, old_mfu)
            else:
                interp_util = old_mfu[0]
                
            lat = row['new_sol_math'] / interp_util if interp_util > 0 else row['new_sol_math']
            
        # 安全守卫：生成的延迟绝对不能低于老款硬件中测出的最短 Kernel 启动时间
        lat = max(lat, min_latency)
        new_latencies.append(lat)
        
    df['new_latency'] = new_latencies
    df['device'] = device_name
    df['latency'] = df['new_latency']
    df['step'] = df['step'].fillna(0).astype(int)
    
    # 格式化导出项保持和最初输入的纯净TXT高度一致
    out_columns = [
        'framework', 'version', 'device', 'op_name', 'kernel_source', 
        'mla_dtype', 'kv_cache_dtype', 'num_heads', 'batch_size', 
        'isl', 'tp_size', 'step', 'latency'
    ]
    
    df[out_columns].to_csv(output_txt, index=False)
    print(f"Extrapolated context latency successfully saved to {output_txt}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(script_dir, "output_context_mla_perf.csv")
    output_txt = os.path.join(script_dir, "vr200_context_mla_perf.txt")
    
    # 解析新硬件配置表，如果没有找到VR200，为了演示将直接构建一个默认的虚拟硬件指标（类比H200/VR200水准）
    sys_vr200_path = os.path.join(script_dir, "..", "vr200.yaml")
    
    if os.path.exists(sys_vr200_path):
        sys_new = load_yaml(sys_vr200_path)
        print(f"Loaded VR200 hardware configs from {sys_vr200_path}")
    else:
        print("VR200 yaml config not found locally. Using mocked default hardware configs for VR200 (6TB/s, 1200 TFLOPS).")
        sys_new = {
            'gpu': {
                'mem_bw': 6 * 1024**4, # 实际可能直接按十进制 6000000000000，这里按严格估算
                'float16_tc_flops': 1200000000000000
            }
        }
        # 为了精确，还是用十进制TB/s更好，这与项目中GB300.yaml 8T等值匹配
        sys_new['gpu']['mem_bw'] = 6000000000000
    
    if os.path.exists(input_csv):
        process_extrapolate_context(sys_new, input_csv, output_txt, device_name="NVIDIA VR200")
    else:
        print(f"Error: Input enriched CSV not found: {input_csv}. Please run data_transfer_attn.py first.")

if __name__ == '__main__':
    main()
