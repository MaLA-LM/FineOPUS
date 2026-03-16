from __future__ import annotations

from create_spreadsheet.aggregate import aggregate_checkpoints_to_csv
from create_spreadsheet.cli import parse_args


def main() -> None:
    args = parse_args()
    stats = aggregate_checkpoints_to_csv(args.results_path, args.output)

    print(f"checkpoint_files_found={stats.checkpoint_files_found}")
    print(f"rows_written={stats.rows_written}")
    print(f"invalid_json_lines_skipped={stats.invalid_json_lines_skipped}")


if __name__ == "__main__":
    main()
