#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import argparse
import sys
import traceback

from huggingface_hub import HfApi, CommitOperationDelete


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def is_same_lang(pair: str) -> bool:
    """True if the folder is like 'src-tgt' with src == tgt (full code, e.g. eng_Latn-eng_Latn)."""
    parts = pair.split("-")
    return len(parts) == 2 and parts[0] == parts[1]


def has_zyyy(pair: str) -> bool:
    """True if either side uses the Zyyy (undetermined) script."""
    return "Zyyy" in pair


def main():
    ap = argparse.ArgumentParser(
        description="Delete top-level language-pair folders from a HF dataset repo "
        "where src==tgt (same full code) or the script is Zyyy."
    )
    ap.add_argument("--repo_id", required=True, help="Target HF dataset repo, e.g. 'MaLA-LM/FineOPUS-Deduplicated'")
    ap.add_argument("--revision", default="main", help="Target branch (default: main)")
    ap.add_argument("--token", default=None, help="Optional HF token (or use env HF_TOKEN)")
    ap.add_argument("--batch_size", type=int, default=100, help="Folders deleted per commit (default: 100)")
    ap.add_argument("--dry_run", action="store_true", help="List folders that would be deleted, but do not delete")
    args = ap.parse_args()

    api = HfApi(token=args.token)

    # Collect the set of top-level folder names that actually exist in the repo.
    logging.info(f"Fetching repo file list for {args.repo_id} ...")
    top_level = set()
    try:
        for f in api.list_repo_files(args.repo_id, repo_type="dataset", revision=args.revision):
            head = f.split("/", 1)[0]
            top_level.add(head)
    except Exception as e:
        logging.error(f"Could not fetch repo file list: {e}")
        sys.exit(1)

    logging.info(f"Found {len(top_level)} top-level entries in repo.")

    # Decide which folders to delete.
    to_delete = sorted(
        d for d in top_level if "-" in d and (is_same_lang(d) or has_zyyy(d))
    )
    n_same = sum(1 for d in to_delete if is_same_lang(d))
    n_zyyy = sum(1 for d in to_delete if has_zyyy(d) and not is_same_lang(d))

    logging.info(f"{len(to_delete)} folders matched (same-code={n_same}, zyyy={n_zyyy}).")

    if args.dry_run:
        for d in to_delete[:60]:
            logging.info(f"[DRY] would delete folder: {d}")
        if len(to_delete) > 60:
            logging.info(f"... (total {len(to_delete)} folders)")
        return

    if not to_delete:
        logging.info("Nothing to delete.")
        return

    ops = [CommitOperationDelete(path_in_repo=f"{d}/", is_folder=True) for d in to_delete]

    BATCH_SIZE = args.batch_size
    total_batches = (len(ops) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(total_batches):
        batch = ops[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        logging.info(f"\n=== Batch {i+1}/{total_batches}: deleting {len(batch)} folders ===")
        try:
            api.create_commit(
                repo_id=args.repo_id,
                repo_type="dataset",
                operations=batch,
                commit_message=f"Delete same-language and Zyyy-script folders {i+1}/{total_batches}",
                revision=args.revision,
            )
            logging.info(f"[OK] Batch {i+1}/{total_batches} deleted successfully.")
        except Exception as e:
            logging.error(f"[ERROR] Batch {i+1} failed: {e}")
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            if batch:
                logging.error(f"First folder in failed batch: {batch[0].path_in_repo}")

    logging.info("\n✅ Deletion complete.")


if __name__ == "__main__":
    main()
