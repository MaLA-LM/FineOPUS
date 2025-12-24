import os
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser(description="Split files into balanced shards based on file size")
    parser.add_argument("--source_dir", type=str, required=True, help="Source directory containing parquet files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for filelists")
    parser.add_argument("--num_splits", type=int, default=512, help="Number of shards to split into (default: 512)")
    parser.add_argument("--relpath", action="store_true", help="Use relative paths instead of absolute paths")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = sorted(glob(f"{args.source_dir}/**/*.parquet", recursive=True))
    file_sizes = [(path, os.path.getsize(path)) for path in all_files]

    file_sizes.sort(key=lambda x: x[1], reverse=True)

    # Allocate the large files to the shard with the lightest load first
    partitions = [[] for _ in range(args.num_splits)]
    partition_loads = [0] * args.num_splits

    for path, size in file_sizes:
        idx = partition_loads.index(min(partition_loads))
        if args.relpath:
            partitions[idx].append(os.path.relpath(path, args.source_dir))
        else:
            partitions[idx].append(path)
        partition_loads[idx] += size

    for i, chunk in enumerate(partitions):
        with open(os.path.join(args.output_dir, f"filelist_{i}.txt"), "w", encoding="utf-8") as f:
            for path in chunk:
                f.write(path + "\n")

    print(f"Created {args.num_splits} filelists in {args.output_dir}")
    print(f"Total files: {len(all_files)}")


if __name__ == "__main__":
    main()
