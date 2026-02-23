### Worker runs

```bash
# 1) create manifest with deterministic shard_id
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 1 
  --out flores200_directions.tsv

# 20 translation directions, JSONL parts rotating by directions/bytes
# --max-seconds-per-part is accepted for CLI compatibility but ignored by JSONL staging.
# 2) run slurm array workers (shard is inferred from Slurm env)
sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=25,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model xcomet

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=20,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model metricx24

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=1,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model qwen

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=20,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model remedy

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=30,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model bicleaner

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=20,TARGET_PART_BYTES=67108864 scripts/run_slurm.sh --manifest flores200_directions.tsv --model Unbabel/M-Prometheus-7B

# 3) optional manual non-array run
python -m src.score_comet 
  --dataset flores200 
  --root /scratch/project_2008161/downstream_benchmarks/flores200 
  --manifest flores200_directions.tsv 
  --worker 
  --output-base /scratch/project_2008161/QE_flores200_scores 
  --model xcomet 
  --num-shards 8 
  --shard-id 0

# 4) compact stage JSONL files into bucketed final parquet output
python -m compact 
  --output-base /scratch/project_2008161/QE_flores200_scores 
  --dataset flores200 
  --num-buckets 2
```
