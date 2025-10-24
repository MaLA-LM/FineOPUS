# To run this script, you first need to install the 'huggingface_hub' library:
# pip install huggingface_hub
#
# You also need to be authenticated. The recommended way is to log in
# via the command line, which securely stores your token:
# huggingface-cli login
#
# --- How to run this script from the command line ---
# To upload:
# python upload_dataset_to_huggingface.py --repo-id "your-username/your-repo-name" --folder-path "/path/to/your/data"
#
# To overwrite and clean the repo first:
# python upload_dataset_to_huggingface.py --repo-id "your-username/your-repo-name" --folder-path "/path/to/converted/data" --clean-repo

import os
import argparse
from huggingface_hub import HfApi, HfFolder

def upload_folder_to_repository(repo_id: str, folder_path: str, clean_repo: bool, squash_history: bool):
    """
    Creates a repository, optionally cleans it, and uploads a local folder.

    Args:
        repo_id (str): The ID of the repository, in "username/repo_name" format.
        folder_path (str): The local path to the folder to upload.
        clean_repo (bool): If True, deletes all files in the repo before uploading.
    """
    print(f"Preparing to upload contents of '{folder_path}' to repository: {repo_id}")


    # --- Step 1: Check if the local folder exists ---
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found. Please provide a valid path.")
        return

    # --- Step 2: Instantiate the HfApi client ---
    api = HfApi()

    # --- Step 3: Create the repository on the Hub ---
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
        )
        print(f"Repository created or already exists: {repo_url}")
    except Exception as e:
        print(f"\n❌ Could not create repository. Error: {e}")
        return

    # --- Step 4 (Optional): Clean the repository ---
    if clean_repo:
        print("\n--clean-repo flag set. Deleting all existing files from the repository...")
        try:
            api.delete_files(
                repo_id=repo_id,
                delete_patterns="*.jsonl",
                repo_type="dataset",
            )
            print("✅ All existing jsonl files deleted successfully.")
        except Exception as e:
            print(f"❌ Could not delete existing files. Error: {e}")
            # Decide if you want to stop or continue if deletion fails
            return

    # --- Step 5: Upload the specified folder ---
    print(f"\nUploading the entire folder from '{folder_path}'...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Folder '{folder_path}' uploaded successfully.")
    except Exception as e:
        print(f"❌ Folder upload failed. Error: {e}")

    # --- Step 6 (Optional): Squash the repository's history ---
    if squash_history:
        print("\n--squash-history flag set. Squashing repository history...")
        try:
            api.super_squash_history(repo_id=repo_id, repo_type="dataset")
            print("✅ History squashed successfully. The repo is now a clean mirror of the local folder.")
        except Exception as e:
            print(f"❌ Could not squash repository history. Error: {e}")
            
    print(f"\n🎉 Operation complete! Check your repository at: {repo_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a local folder to a Hugging Face Hub dataset repository."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="The ID of the repository on the Hub, e.g., 'your-username/your-dataset-name'."
    )
    parser.add_argument(
        "--folder-path",
        type=str,
        required=True,
        help="The path to the local folder containing the data you want to upload."
    )
    parser.add_argument(
        "--clean-repo",
        action="store_true",
        help="If set, this flag will delete all files in the repository before uploading."
    )

    parser.add_argument(
        "--squash-history",
        action="store_true",
        help="If set, squashes the repo's entire git history after uploading. "
             "This makes the repo a clean, single-commit mirror of the local folder."
    )
    
    args = parser.parse_args()
    
    upload_folder_to_repository(
        repo_id=args.repo_id, 
        folder_path=args.folder_path,
        clean_repo=args.clean_repo,
        squash_history=args.squash_history
    )

