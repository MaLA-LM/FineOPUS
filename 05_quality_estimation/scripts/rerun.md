# OPUS MetricX Reruns

These commands target the unfinished `metricx24` manifest workers from
`opus-manifest-2026-05-05`.

Use `OPUS_ARRAY_TASK_ID` / `OPUS_LOCAL_ID` for targeted single-GCD reruns.
Do not use `SLURM_LOCALID` for this purpose; Slurm can overwrite it inside an
`sbatch` job.

All commands use `BATCH_SIZE=32` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce the MetricX OOM
risk seen with `BATCH_SIZE=64`.

## Dense Nodes

Arrays with more than three unfinished workers are rerun as whole standard-g
nodes. This covers arrays `16,31,32,33,34,35,36,48,63,79,96`.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/opus/submit_array_standard_g.sh --account project_462001249 --model metricx24 --array 16,31,32,33,34,35,36,48,63,79,96 --concurrency 11 --time 47:00:00 --batch-size 32 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3



bash scripts/opus/submit_array_standard_g.sh --account <> --model shaomutan_remedy-9b-22 --array 37,39,13,12 --concurrency 4 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3

bash scripts/opus/submit_array_standard_g.sh --account <> --model qwen3-4b-instruct-2507 --array 72,73 --concurrency 2 --time 47:00:00 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3  --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 75000 --response-format json_schema --enforce-eager --part-writer --part-max-bytes 5536870912 --part-max-shards 50

```

## Sparse Nodes

No array has fewer than three unfinished workers in the supplied list. Array
`37` has exactly three unfinished workers, so it is cheaper to rerun those
slots individually on small-g instead of reserving a whole standard-g node.

```bash
sbatch --array=84-84 --account=<> --partition=small-g --time=47:00:00 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --export=ALL,MODEL=metricx24,MANIFEST_ROOT=/scratch/project_462001069/opus_qe/manifests,BUILD_TAG=opus-manifest-2026-05-05,TRACE_ROOT=/scratch/project_462001069/opus_qe/shard_trace,OUTPUT_BASE=/scratch/project_462001069/opus_qe/shards,OPUS_ROOT=/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3,OPUS_ARRAY_TASK_ID=37,OPUS_LOCAL_ID=0,BATCH_SIZE=32,PART_WRITER=1,PART_MAX_BYTES=5536870912,PART_MAX_SHARDS=50,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/opus/run_worker.sh
```

```bash
sbatch --array=37-37 --account=project_462001249 --partition=small-g --time=40:00:00 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --export=ALL,MODEL=metricx24,MANIFEST_ROOT=/scratch/project_462001069/opus_qe/manifests,BUILD_TAG=opus-manifest-2026-05-05,TRACE_ROOT=/scratch/project_462001069/opus_qe/shard_trace,OUTPUT_BASE=/scratch/project_462001069/opus_qe/shards,OPUS_ROOT=/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3,OPUS_ARRAY_TASK_ID=37,OPUS_LOCAL_ID=1,BATCH_SIZE=32,PART_WRITER=1,PART_MAX_BYTES=5536870912,PART_MAX_SHARDS=50,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/opus/run_worker.sh
```

```bash
sbatch --array=37-37 --account=project_462001249 --partition=small-g --time=40:00:00 --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=60G --export=ALL,MODEL=metricx24,MANIFEST_ROOT=/scratch/project_462001069/opus_qe/manifests,BUILD_TAG=opus-manifest-2026-05-05,TRACE_ROOT=/scratch/project_462001069/opus_qe/shard_trace,OUTPUT_BASE=/scratch/project_462001069/opus_qe/shards,OPUS_ROOT=/scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3,OPUS_ARRAY_TASK_ID=37,OPUS_LOCAL_ID=2,BATCH_SIZE=32,PART_WRITER=1,PART_MAX_BYTES=5536870912,PART_MAX_SHARDS=50,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True scripts/opus/run_worker.sh
```

## Check Progress

```bash
python -m stand_alone_modules.opus_trace_summary --model metricx24 --trace-root /scratch/project_462001069/opus_qe/shard_trace --build-tag opus-manifest-2026-05-05 --manifest-root /scratch/project_462001069/opus_qe/manifests
```

## remedy
```bash

OPUS_ARRAY_TASK_ID=29 OPUS_LOCAL_ID=5 bash scripts/opus/submit_array.sh --account project_462001249 --partition small-g --model shaomutan_remedy-9b-22 --array 29 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3 

## qwen
OPUS_ARRAY_TASK_ID=84 OPUS_LOCAL_ID=7 bash scripts/opus/submit_array.sh --account <> --partition small-g --model qwen3-4b-instruct-2507 --array 84 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3 --batch-size 32 --prompt-mode batch --max-tokens 8192 --max-num-batched-tokens 8192 --max-num-seqs 32 --max-model-len 75000 --response-format json_schema --enforce-eager

### to be run

OPUS_ARRAY_TASK_ID=24 OPUS_LOCAL_ID=1 bash scripts/opus/submit_array.sh --account <> --partition small-g --model shaomutan_remedy-9b-22 --array 24 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3

OPUS_ARRAY_TASK_ID=32 OPUS_LOCAL_ID=3 bash scripts/opus/submit_array.sh --account <> --partition small-g --model shaomutan_remedy-9b-22 --array 32 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3

OPUS_ARRAY_TASK_ID=30 OPUS_LOCAL_ID=4 bash scripts/opus/submit_array.sh --account <> --partition small-g --model shaomutan_remedy-9b-22 --array 30 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3

OPUS_ARRAY_TASK_ID=24 OPUS_LOCAL_ID=2 bash scripts/opus/submit_array.sh --account <> --partition small-g --model shaomutan_remedy-9b-22 --array 24 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3

OPUS_ARRAY_TASK_ID=24 OPUS_LOCAL_ID=0 bash scripts/opus/submit_array.sh --account <> --partition small-g --model shaomutan_remedy-9b-22 --array 24 --concurrency 1 --time 47:00:00 --part-writer --part-max-bytes 5536870912 --part-max-shards 50 --manifest-root /scratch/project_462001069/opus_qe/manifests --build-tag opus-manifest-2026-05-05 --trace-root /scratch/project_462001069/opus_qe/shard_trace --output-base /scratch/project_462001069/opus_qe/shards --opus-root /scratch/project_462001249/MaLA-LM/FineOPUS-Filtered-Stage3





```


