### Worker runs

```bash
# 1) create manifest with deterministic shard_id
python -m execution.flores_array.make_manifest --dataset flores200 --split all --num-shards 1   --out flores200_directions.tsv

---

# experiments
# bicleaner-> 
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 15 
  --out flores200_directions_bicleaner.tsv

sbatch --array=0-14 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions_bicleaner.tsv --model bicleaner

---

# metricx-24-> 
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 35 
  --out flores200_directions_metricx.tsv  

sbatch --array=0 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_metricx.tsv --model metricx24 --num-shards 35

sbatch --array=0-$((N-1)) --export=ALL,HF_TOKEN=$HF_TOKEN,BATCH_SIZE=64,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model metricx24

---

# remedy-> 
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 104
  --out flores200_directions_remedy.tsv  

sbatch --array=0-103 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions_remedy.tsv --model remedy

---

# to be submitted
# xcomet-> 
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 57
  --out flores200_directions_xcomet.tsv  

sbatch --array=0-56 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions_xcomet.tsv --model xcomet   

---

# comet23-> 
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 70
  --out flores200_directions_comet23.tsv   

sbatch --array=0-69 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions_comet23.tsv --model comet23
 
# M-prometheus->
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 180 
  --out flores200_directions_prometheus.tsv

sbatch --array=0-179 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_prometheus.tsv --model m-prometheus-7b


# Qwen prompts
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 90
  --out flores200_directions_qwen4b.tsv
# normal prompt
sbatch --array=0-89 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b --num-shards 90
# simpler prompt
sbatch --array=0-89 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728,PROMPT_MODE=simple scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b --num-shards 90


## qwen3-4b-fp8, qwen3-4b-instruct-2507-fp8, qwen3-4b-awq
sbatch --array=0 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b-awq --num-shards 90

# Test batch mode (batch_size=8 by default)
sbatch --array=0 --time=01:00:00 --export=ALL,HF_TOKEN=$HF_TOKEN,PROMPT_MODE=batch scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b --num-shards 90

# replicating old results exactly for LLMs
sbatch --array=0 --time=01:00:00 --export=ALL,HF_TOKEN=$HF_TOKEN,PROMPT_MODE=batch,BATCH_SIZE=32,RESPONSE_FORMAT=json_schema,MAX_NUM_BATCHED_TOKENS=8192,MAX_NUM_SEQS=32,ENFORCE_EAGER=1,MAX_TOKENS=8192 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b --num-shards 90



# Match old vLLM-server settings: json_schema (default), same vLLM scheduler params
# --response-format options: none (fast, no constraint), json_object (valid JSON), json_schema (per-token schema enforcement, old default)
sbatch --array=0 --time=01:00:00 --export=ALL,HF_TOKEN=$HF_TOKEN,PROMPT_MODE=batch,BATCH_SIZE=16,MAX_NUM_BATCHED_TOKENS=8192,MAX_NUM_SEQS=32,MAX_MODEL_LEN=8192 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen3.tsv --model qwen3-4b --num-shards 90




---

# qwen3-4B->
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 190
  --out flores200_directions_qwen4b.tsv

sbatch --array=0-189 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen4b.tsv --model qwen3-4b


# qwen3-8b->
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 190
  --out flores200_directions_qwen8b.tsv

sbatch --array=0-189 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen8b.tsv --model qwen3-8b

# qwen3-14b->
python -m execution.flores_array.make_manifest 
  --dataset flores200
  --split all 
  --num-shards 190
  --out flores200_directions_qwen14b.tsv

sbatch --array=0-189 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions_qwen14b.tsv --model qwen3-14b
---

# 20 translation directions, JSONL parts rotating by directions/bytes
# --max-seconds-per-part is accepted for CLI compatibility but ignored by JSONL staging.
# 2) run slurm array workers (shard is inferred from Slurm env)
sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model xcomet

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model metricx24

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model qwen

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model remedy

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model bicleaner

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm.sh --manifest flores200_directions.tsv --model Unbabel/M-Prometheus-7B

---
#LUMI tests

# LUMI ReMedy strict path (exact vLLM 0.9.2 ROCm):
bash scripts/build_container_lumi.sh source

REMEDY_SIF=/scratch/project_462001050/$USER/envs/remedy_source.sif
singularity exec --rocm "$REMEDY_SIF" python -c "import vllm, torch; assert vllm.__version__ == '0.9.2'; assert getattr(torch.version, 'hip', None); print(vllm.__version__, torch.version.hip)"

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model xcomet

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model metricx24

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model qwen

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model remedy

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model bicleaner

sbatch --array=1-1 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=100,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model Unbabel/M-Prometheus-7B

sbatch --array=0 --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 scripts/flores/run_slurm_lumi.sh --manifest flores200_directions.tsv --model m-prometheus-7b --num-shards 1


---

# reruns
sbatch --array=81,128 
  --export=ALL,HF_TOKEN=$HF_TOKEN,MAX_DIRECTIONS_PER_PART=200,TARGET_PART_BYTES=134217728 
  scripts/flores/run_slurm_lumi.sh 
  --manifest flores200_directions_qwen4b.tsv 
  --model qwen3-4b 
  --num-shards 190

---
# 3) optional manual non-array run
python -m src.backends.comet 
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
  --output-base /scratch/project_462001050/QE_flores200_scores
  --dataset flores200
  --num-buckets 2


# 5) check done, model name as directory name
# models: shaomutan_remedy-9b-22, wmt23-cometkiwi-da-xl, xcomet-xl, metricx24, m-prometheus-7b, qwen3-0.6b
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_bicleaner.tsv
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_qwen3.tsv --model qwen3-1.7b --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_qwen14b.tsv --model qwen3-14b --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_remedy.tsv --model shaomutan_remedy-9b-22 --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200
python -m stand_alone_modules.check_done.check_shards --tsv flores200_directions_qwen3.tsv --model qwen3-4b-instruct-2507-simple --path /scratch/project_462001050/QE_flores200_scores/dataset=flores200


# 6) csv summary
python -m stand_alone_modules.create_spreadsheet /scratch/project_462001050/QE_flores200_scores/dataset=flores200 --output flores200_benchmark_results_1.csv

# 7) dedup
python -m stand_alone_modules.dedup scan --dataset-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200 --output .
python -m stand_alone_modules.dedup apply --plan /scratch/.../dataset=flores200/dedup_plan.json

#8) patch results seen/unseen + null scores
# Step 1: patch (creates *-patched.jsonl files alongside originals)
python -m stand_alone_modules.patch_results patch --model-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200/model=m-prometheus-7b
# Step 2: replace (swaps patched files into original names, deletes old)
python -m stand_alone_modules.patch_results replace --model-path /scratch/project_462001050/QE_flores200_scores/dataset=flores200/model=m-prometheus-7b

# 9) compcation:
singularity run $SIF python -m stand_alone_modules.compact
  --output-base /scratch/project_462001050/QE_flores200_scores
  --dataset flores200
  --name raw_scores_new
  --target-part-bytes 671088640
  --workers 6

```

