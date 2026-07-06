from pathlib import Path
import torch
from collector.sglang.collect_mla import run_mla

out = Path('/workspace/.self/task-localbench/tmp/tmp-collect-mla-gen/generation_mla_perf_resample.txt')
out.unlink(missing_ok=True)
points = [
    # s_total, batch_size, label
    (512, 1, 'b1_s512'),
    (512, 4, 'b4_s512'),
    (512, 32, 'b32_s512'),
    (4096, 4, 'b4_s4096'),
    (16384, 2, 'b2_s16384'),
]
for s_total, batch_size, label in points:
    print(f'RUN {label}', flush=True)
    run_mla(
        s_total - 1,
        batch_size,
        1,
        torch.float8_e4m3fn,
        128,
        1,
        1,
        64,
        3,
        20,
        False,
        perf_filename=str(out),
        device='cuda:0',
    )
print(f'WROTE {out}', flush=True)
