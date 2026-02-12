
### test runs

```bash

1) sbatch --export=ALL,HF_TOKEN=$HF_TOKEN scripts/run_slurm.sh --mode worker --model Qwen/Qwen3-14B --worker-max-files 2 --manifest flores200_directions.tsv
2) sbatch --array=1-2 --export=ALL,HF_TOKEN=$HF_TOKEN scripts/run_slurm.sh --mode worker --model xcomet --worker-max-files 2 --manifest flores200_directions.tsv
3) sbatch --export=ALL,HF_TOKEN=$HF_TOKEN scripts/run_slurm.sh --mode worker --model metricx24 --worker-max-files 2 --manifest flores200_directions.tsv
4) sbatch  --export=ALL,HF_TOKEN=$HF_TOKEN scripts/run_slurm.sh --mode worker --model bicleaner-ai --worker-max-files 2 --manifest flores200_directions.tsv

for tokens:
sbatch --export=ALL,HF_TOKEN=$HF_TOKEN scripts/run_slurm.sh


## manifest

python -m dataset.make_manifest --dataset flores200 --split all --out flores200_directions.tsv

```
