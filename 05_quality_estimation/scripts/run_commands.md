### Worker runs

```bash
# 1) create manifest with deterministic shard_id
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 1 
  --out flores200_directions.tsv

---

# experiments
# bicleaner-> 
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 15 
  --out flores200_directions_bicleaner.tsv

sbatch --array=0-14 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions_bicleaner.tsv --model bicleaner

---

# metricx-24-> 
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 35 
  --out flores200_directions_metricx.tsv  

sbatch --array=0-34 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions_metricx.tsv --model metricx24  

---

# remedy-> 
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 104
  --out flores200_directions_remedy.tsv  

sbatch --array=0-103 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions_remedy.tsv --model remedy

---

# to be submitted
# xcomet-> 
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 57
  --out flores200_directions_xcomet.tsv  

sbatch --array=0-56 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions_xcomet.tsv --model xcomet   

---

# comet23-> 
python -m dataset.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 70
  --out flores200_directions_comet23.tsv   

sbatch --array=0-69 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions_comet23.tsv --model comet23
 

---

# 20 translation directions, JSONL parts rotating by directions/bytes
# --max-seconds-per-part is accepted for CLI compatibility but ignored by JSONL staging.
# 2) run slurm array workers (shard is inferred from Slurm env)
sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model xcomet

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model metricx24

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model qwen

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model remedy

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model bicleaner

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/run_slurm.sh --manifest flores200_directions.tsv --model Unbabel/M-Prometheus-7B

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
