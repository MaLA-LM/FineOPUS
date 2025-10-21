#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, create_repo
import sys
from tqdm import tqdm
import traceback


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def main():
    ap = argparse.ArgumentParser(description="Upload a large dataset folder to Hugging Face Hub.")
    ap.add_argument("--base_path", required=True, help="Local base directory to upload")
    ap.add_argument("--repo_id", required=True, help="Target Hugging Face dataset repo, e.g., 'Helsinki-NLP/xxx'")
    ap.add_argument("--path_in_repo", default="", help="Optional subdirectory path inside repo")
    ap.add_argument("--revision", default="main", help="Target branch (default: main)")
    ap.add_argument("--token", default=None, help="Optional HF token (or use env HF_TOKEN)")
    ap.add_argument("--batch_size", type=int, default=100, help="Batch size for uploading (default: 100)")
    ap.add_argument("--dry_run", action="store_true", help="List planned uploads only")
    args = ap.parse_args()

    base_path = Path(args.base_path).resolve()
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    api = HfApi(token=args.token)
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    # Get existing files list (avoid duplicate uploads)
    logging.info(f"Fetching repo file list for {args.repo_id} ...")
    existing_files = set()
    try:
        for f in api.list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision):
            existing_files.add(f)
    except Exception as e:
        logging.warning(f"[WARN] Could not fetch repo file list: {e}")

    logging.info(f"Found {len(existing_files)} existing files in repo.")

    # Scan local files
    local_files = []
    for root, _, files in os.walk(base_path):
        for fname in files:
            fpath = Path(root) / fname
            rel_path = str(fpath.relative_to(base_path)).replace("\\", "/")
            if args.path_in_repo:
                rel_path = f"{args.path_in_repo.rstrip('/')}/{rel_path}"
            local_files.append((fpath, rel_path))

    logging.info(f"Scanning done: {len(local_files)} local files found.")

    # Only upload missing files
    upload_ops = []
    for local_path, rel_path in tqdm(local_files, desc="Checking files"):
        if rel_path not in existing_files:
            upload_ops.append(CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=str(local_path)))

    logging.info(f"{len(upload_ops)} new files to upload.")
    if args.dry_run:
        for op in upload_ops[:30]:
            logging.info(f"[DRY] {op.path_in_repo}")
        if len(upload_ops) > 30:
            logging.info(f"... (total {len(upload_ops)} files)")
        return

    # Batch upload to prevent large submissions
    BATCH_SIZE = args.batch_size
    total_batches = (len(upload_ops) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(total_batches):
        batch = upload_ops[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        logging.info(f"\n=== Batch {i+1}/{total_batches}: uploading {len(batch)} files ===")
        try:
            api.create_commit(
                repo_id=args.repo_id,
                repo_type="dataset",
                operations=batch,
                commit_message=f"Batch upload {i+1}/{total_batches}",
                revision=args.revision
            )
            logging.info(f"[OK] Batch {i+1}/{total_batches} uploaded successfully.")
        except Exception as e:
            logging.error(f"[ERROR] Batch {i+1} failed: {e}")
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            # Also log the first file in the failed batch for debugging
            if batch:
                logging.error(f"First file in failed batch: {batch[0].path_in_repo}")

    logging.info("\n✅ Upload complete.")

if __name__ == "__main__":
    main()
