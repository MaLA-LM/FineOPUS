from __future__ import annotations


def validate_args(args) -> None:
    if not args.worker:
        raise SystemExit("Pipeline supports worker mode only; pass --worker.")
    if not args.manifest:
        raise SystemExit("--manifest is required.")
    if not args.output_base:
        raise SystemExit("--output-base is required.")

    if (args.num_shards is None) != (args.shard_id is None):
        raise SystemExit("Provide both --num-shards and --shard-id, or neither.")
    if args.num_shards is not None and args.num_shards <= 0:
        raise SystemExit("--num-shards must be > 0.")
    if args.shard_id is not None and args.shard_id < 0:
        raise SystemExit("--shard-id must be >= 0.")
    if (
        args.num_shards is not None
        and args.shard_id is not None
        and args.shard_id >= args.num_shards
    ):
        raise SystemExit("--shard-id must be in [0, --num-shards-1].")

    if args.max_directions_per_part <= 0:
        raise SystemExit("--max-directions-per-part must be > 0.")
    if args.max_seconds_per_part <= 0:
        raise SystemExit("--max-seconds-per-part must be > 0.")
    if args.target_part_bytes <= 0:
        raise SystemExit("--target-part-bytes must be > 0.")
