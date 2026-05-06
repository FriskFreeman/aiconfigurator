import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = '/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/workshop/output_generation_mla_perf.csv'
out_dir = '/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/workshop/plot'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Ensure numeric types
df['step'] = pd.to_numeric(df['step'])
df['intensity_op'] = pd.to_numeric(df['intensity_op'])
df['mbu'] = pd.to_numeric(df['mbu'])

num_heads_list = df['num_heads'].unique()

for num_heads in num_heads_list:
    df_head = df[df['num_heads'] == num_heads]
    
    # We might have different kv_cache_dtype and tp_size
    combinations = df_head[['mla_dtype', 'kv_cache_dtype', 'tp_size']].drop_duplicates()
    
    for _, row in combinations.iterrows():
        mla_dtype = row['mla_dtype']
        kv_cache_dtype = row['kv_cache_dtype']
        tp_size = row['tp_size']
        
        df_sub = df_head[(df_head['mla_dtype'] == mla_dtype) & 
                         (df_head['kv_cache_dtype'] == kv_cache_dtype) &
                         (df_head['tp_size'] == tp_size)]
        
        # Plot 1: step vs mbu
        plt.figure(figsize=(10, 6))
        for batch_size in sorted(df_sub['batch_size'].unique()):
            df_bs = df_sub[df_sub['batch_size'] == batch_size].sort_values('step')
            plt.plot(df_bs['step'], df_bs['mbu'], marker='o', label=f'BS={batch_size}')
        
        plt.xscale('log', base=2)
        plt.xlabel('Step (log2 scale)')
        plt.ylabel('MBU')
        plt.title(f'Step vs MBU\n(num_heads={num_heads}, mla={mla_dtype}, kv={kv_cache_dtype}, tp={tp_size})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'step_vs_mbu_heads{num_heads}_mla{mla_dtype}_kv{kv_cache_dtype}_tp{tp_size}.png'))
        plt.close()
        
        # Plot 2: intensity_op vs mbu
        plt.figure(figsize=(10, 6))
        for batch_size in sorted(df_sub['batch_size'].unique()):
            df_bs = df_sub[df_sub['batch_size'] == batch_size].sort_values('intensity_op')
            plt.plot(df_bs['intensity_op'], df_bs['mbu'], marker='o', label=f'BS={batch_size}')
            
        plt.xlabel('Intensity OP')
        plt.ylabel('MBU')
        plt.title(f'Intensity OP vs MBU\n(num_heads={num_heads}, mla={mla_dtype}, kv={kv_cache_dtype}, tp={tp_size})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'intensity_vs_mbu_heads{num_heads}_mla{mla_dtype}_kv{kv_cache_dtype}_tp{tp_size}.png'))
        plt.close()

print(f"Plots generated successfully in {out_dir}")
