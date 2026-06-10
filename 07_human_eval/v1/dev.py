from datasets import get_dataset_config_names

repo_id = "MaLA-LM/FineOPUS-Original"

# This fetches only the metadata, not the actual data files
# configs = get_dataset_config_names(repo_id)

# print(f"Found {len(configs)} language pairs:")
# for config in configs[:10]:  # Show first 10
#     print(f" - {config}")

from huggingface_hub import list_repo_files
repo_id = "MaLA-LM/FineOPUS-Original"
files = list_repo_files(repo_id, repo_type="dataset")
print(files[:10])