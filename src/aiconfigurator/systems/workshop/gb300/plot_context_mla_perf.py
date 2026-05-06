import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = '/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/workshop/gb300/output_context_mla_perf.csv'
out_dir = '/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/workshop/gb300/plot_context'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Ensure numeric types
df['isl'] = pd.to_numeric(df['isl'])
df['intensity_op'] = pd.to_numeric(df['intensity_op'])
df['mbu'] = pd.to_numeric(df['mbu'])
df['mfu'] = pd.to_numeric(df['mfu'])

num_heads_list = df['num_heads'].unique()

for num_heads in num_heads_list:
    df_head = df[df['num_heads'] == num_heads]
    
    combinations = df_head[['mla_dtype', 'kv_cache_dtype', 'tp_size']].drop_duplicates()
    
    for _, row in combinations.iterrows():
        mla_dtype = row['mla_dtype']
        kv_cache_dtype = row['kv_cache_dtype']
        tp_size = row['tp_size']
        
        df_sub = df_head[(df_head['mla_dtype'] == mla_dtype) & 
                         (df_head['kv_cache_dtype'] == kv_cache_dtype) &
                         (df_head['tp_size'] == tp_size)]
        
        metrics = ['mbu', 'mfu']
        x_axes = [('isl', True), ('intensity_op', False)]
        
        for y_metric in metrics:
            for x_metric, is_log in x_axes:
                plt.figure(figsize=(10, 6))
                for batch_size in sorted(df_sub['batch_size'].unique()):
                    df_bs = df_sub[df_sub['batch_size'] == batch_size].sort_values(x_metric)
                    
                    if y_metric == 'mbu':
                        df_bs = df_bs[df_bs['intensity_op'] <= 1000]
                    elif y_metric == 'mfu':
                        df_bs = df_bs[df_bs['intensity_op'] >= 200]
                        
                    if not df_bs.empty:
                        plt.plot(df_bs[x_metric], df_bs[y_metric], marker='o', label=f'BS={batch_size}')
                
                if x_metric == 'isl':
                    vline_x = 312.5 * 3 if kv_cache_dtype == 'fp8' else 312.5 * 4
                    plt.axvline(x=vline_x, color='grey', linestyle='--', label=f'Helper (x={vline_x})')
                elif x_metric == 'intensity_op':
                    plt.axvline(x=312.5, color='grey', linestyle='--', label='Helper (x=312.5)')

                if is_log:
                    plt.xscale('log', base=2)
                    plt.xlabel(f'{x_metric.capitalize()} (log2 scale)')
                else:
                    plt.xlabel(f'{x_metric.capitalize()}')
                    
                plt.ylabel(y_metric.upper())
                plt.title(f'{x_metric} vs {y_metric.upper()}\n(num_heads={num_heads}, mla={mla_dtype}, kv={kv_cache_dtype}, tp={tp_size})')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'{x_metric}_vs_{y_metric}_heads{num_heads}_mla{mla_dtype}_kv{kv_cache_dtype}_tp{tp_size}.png'))
                plt.close()

print(f"Plots generated successfully in {out_dir}")
