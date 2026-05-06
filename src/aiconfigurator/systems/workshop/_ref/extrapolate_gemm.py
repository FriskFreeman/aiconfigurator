import pandas as pd
import yaml
import numpy as np
import os

def get_bytes_per_elem(dtype):
    if dtype in ['float16', 'fp16']: return 2.0
    if dtype in ['fp8', 'int8']: return 1.0
    if dtype in ['nvfp4', 'fp4']: return 0.5
    raise ValueError(f"Unknown dtype {dtype}")

def get_tc_flops(yaml_data, dtype):
    gpu = yaml_data['gpu']
    if dtype in ['float16', 'fp16']: return gpu['float16_tc_flops']
    if dtype in ['fp8']: return gpu['fp8_tc_flops']
    if dtype in ['int8']: return gpu['int8_tc_flops']
    if dtype in ['nvfp4', 'fp4']: return gpu['fp4_tc_flops']
    raise ValueError(f"Unknown dtype {dtype}")

def process_gemm(gb300_sys, vr200_sys):
    gb300_perf_path = 'src/aiconfigurator/systems/data/gb300/sglang/0.5.9/gemm_perf.txt'
    vr200_out_dir = 'my_systems/data/vr200/sglang/0.5.9'
    os.makedirs(vr200_out_dir, exist_ok=True)
    vr200_perf_path = os.path.join(vr200_out_dir, 'gemm_perf.txt')

    print(f"Loading GB300 data from {gb300_perf_path}")
    df = pd.read_csv(gb300_perf_path)
    
    # Vectorized operations for GB300
    df['bytes_elem'] = df['gemm_dtype'].apply(get_bytes_per_elem)
    df['tc_flops'] = df['gemm_dtype'].apply(lambda d: get_tc_flops(gb300_sys, d))
    mem_bw = gb300_sys['gpu']['mem_bw']
    
    flops = 2 * df['m'] * df['n'] * df['k']
    mem_bytes = df['bytes_elem'] * (df['m'] * df['k'] + df['n'] * df['k']) + 2 * df['bytes_elem'] * (df['m'] * df['n'])
    
    df['sol_math'] = flops / df['tc_flops'] * 1000.0
    df['sol_mem'] = mem_bytes / mem_bw * 1000.0
    
    df['intensity_op'] = flops / mem_bytes
    df['intensity_hw'] = df['tc_flops'] / mem_bw
    df['rel_ratio'] = df['intensity_op'] / df['intensity_hw']
    
    # User's formulas: actual delay / theoretical delay
    df['mfu'] = np.where(df['latency'] > 0, df['sol_math'] / df['latency'], 0)
    df['mbu'] = np.where(df['latency'] > 0, df['sol_mem'] / df['latency'], 0)
    
    print("GB300 processing complete. Sample data (first 5 rows):")
    print(df[['gemm_dtype', 'rel_ratio', 'mfu', 'mbu']].head())
    
    print("\nStarting VR200 Hardware Extrapolation...")
    df_vr200 = df.copy()
    
    df_vr200['tc_flops_vr200'] = df_vr200['gemm_dtype'].apply(lambda d: get_tc_flops(vr200_sys, d))
    mem_bw_vr200 = vr200_sys['gpu']['mem_bw']
    
    df_vr200['sol_math'] = flops / df_vr200['tc_flops_vr200'] * 1000.0
    df_vr200['sol_mem'] = mem_bytes / mem_bw_vr200 * 1000.0
    
    df_vr200['intensity_hw'] = df_vr200['tc_flops_vr200'] / mem_bw_vr200
    df_vr200['rel_ratio'] = df_vr200['intensity_op'] / df_vr200['intensity_hw']
    
    # Group by dtype and check for shape-specific (k,n) matches to improve interpolation accuracy
    for dtype in df_vr200['gemm_dtype'].unique():
        dtype_mask = df_vr200['gemm_dtype'] == dtype
        if not dtype_mask.any(): continue
        
        # Identify unique shapes for this dtype to perform more granular matching
        shapes = df_vr200.loc[dtype_mask, ['k', 'n']].drop_duplicates()
        
        for _, row_shape in shapes.iterrows():
            k_val, n_val = int(row_shape['k']), int(row_shape['n'])
            mask = dtype_mask & (df_vr200['k'] == k_val) & (df_vr200['n'] == n_val)
            
            # Attempt to find exact (k, n) matches in GB300 for the current dtype
            subset_gb300 = df[
                (df['gemm_dtype'] == dtype) & 
                (df['k'] == k_val) & 
                (df['n'] == n_val)
            ].sort_values('rel_ratio')
            
            # Fallback 1: Try swapped dimensions (n, k)
            if subset_gb300.empty:
                subset_gb300 = df[
                    (df['gemm_dtype'] == dtype) & 
                    (df['k'] == n_val) & 
                    (df['n'] == k_val)
                ].sort_values('rel_ratio')
                
            # Fallback 2: General dtype-only distribution
            if subset_gb300.empty:
                subset_gb300 = df[df['gemm_dtype'] == dtype].sort_values('rel_ratio')
            
            if subset_gb300.empty:
                continue
                
            rel_ratio_vals = df_vr200.loc[mask, 'rel_ratio']
            
            mfu_new = np.interp(rel_ratio_vals, subset_gb300['rel_ratio'], subset_gb300['mfu'])
            mbu_new = np.interp(rel_ratio_vals, subset_gb300['rel_ratio'], subset_gb300['mbu'])
            
            df_vr200.loc[mask, 'mfu'] = mfu_new
            df_vr200.loc[mask, 'mbu'] = mbu_new
        
    df_vr200['latency'] = np.where(
        df_vr200['rel_ratio'] > 1.0,
        df_vr200['sol_math'] / df_vr200['mfu'],
        df_vr200['sol_mem'] / df_vr200['mbu']
    )
    
    out_columns = ['framework','version','device','op_name','kernel_source','gemm_dtype','m','n','k','latency']
    df_vr200[out_columns].to_csv(vr200_perf_path, index=False)
    print(f"Saved pure VR200 perf.txt to {vr200_perf_path}")
    
    # Save enriched data with requested logged fields
    df.to_csv('gb300_gemm_perf_enriched.csv', index=False)
    df_vr200.to_csv('my_systems/vr200_gemm_perf_enriched.csv', index=False)
    print("Saved enriched detailed CSVs for GB300 and VR200 GEMM with mfu, mbu, rel_ratio records.\n")

def main():
    gb300_yaml_path = 'src/aiconfigurator/systems/gb300.yaml'
    vr200_yaml_path = 'my_systems/vr200.yaml'
    
    with open(gb300_yaml_path) as f:
        gb300_sys = yaml.safe_load(f)
    with open(vr200_yaml_path) as f:
        vr200_sys = yaml.safe_load(f)
        
    process_gemm(gb300_sys, vr200_sys)

if __name__ == "__main__":
    main()
