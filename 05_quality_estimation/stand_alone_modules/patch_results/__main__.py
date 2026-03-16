from patch_results.cli import parse_args
from utils.logger import logger

args = parse_args()

if args.command == "patch":
    from patch_results.discovery import discover_shards_needed
    from patch_results.patch import patch

    shards = discover_shards_needed(args.model_path)
    if not shards:
        logger.warning("No shards found under %s \u2013 nothing to patch.", args.model_path)
    else:
        patch(shards, max_workers=min(len(shards), 64))

elif args.command == "replace":
    from patch_results.replace import replace

    replace(args.model_path)
