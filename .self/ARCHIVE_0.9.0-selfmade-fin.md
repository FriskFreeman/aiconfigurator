# 0.9.0 self-made experiment archive

This branch preserves the lightweight, reproducible parts of local AIC experiments as of
2026-08-04. It is an archival snapshot rather than a complete copy of every generated run.

## Included

- Experiment notes, reports, scripts, configurations, and analysis utilities under `.self/`
  and `collector/[self]doc/`.
- Compact summary tables, JSON metadata, and generated figures needed to understand the
  conclusions of the analytical-mode and collector comparison experiments.
- One latest representative result for each formal local-benchmark series:
  - aggregate stage 2: `20260630_185819_agg_stage2_agg_aic_prefix_probe_*`
  - decode stage 1: `20260708_105525_prefill_stage1_decode_kvfp8_default_*`
  - prefill stage 1: `20260708_095702_prefill_stage1_kvfp8_probe_*`
  - MLA module quantization: `20260629_122357_module_timing_decode10_*`
- The latest complete AIC-origin PD run (`20260622_155838_*`) and the later MLA collector
  smoke result (`20260626_000001_*`).
- Existing tracked `bench_data` summaries and analysis artifacts inherited from the branch
  history. No raw benchmark run was newly added by this archive commit.

## Excluded

- `.self/sharegpt/`, which contains third-party data.
- Raw profiler databases and captures (`report`, `*.sqlite`, `*.nsys-rep`).
- Full raw runs, checkpoints, caches, temporary experiments, generated request payloads,
  and verbose execution logs.
- Raw Codex web/network event captures under `.self/codex serverinfo/`.
- Unrelated untracked system performance datasets under `src/aiconfigurator/systems/data/`.
- Large compressed point-level datasets (`*.csv.gz`) and duplicate PDF renderings when a
  PNG plus compact CSV summary is available.

Ignored representative files are intentionally force-added to this branch. The general
ignore rules remain unchanged so future raw experiment runs do not enter Git accidentally.
