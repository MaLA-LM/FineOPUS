from __future__ import annotations


def validate_args(args) -> None:
    if not args.manifest:
        raise SystemExit("--manifest is required.")
    if not args.output_base:
        raise SystemExit("--output-base is required.")
    if args.shard_id is not None and args.shard_id < 0:
        raise SystemExit("--shard-id must be >= 0.")
    if args.num_shards is not None and args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0.")
    if args.max_directions_per_part <= 0:
        raise SystemExit("--max-directions-per-part must be > 0.")
    if args.target_part_bytes <= 0:
        raise SystemExit("--target-part-bytes must be > 0.")
