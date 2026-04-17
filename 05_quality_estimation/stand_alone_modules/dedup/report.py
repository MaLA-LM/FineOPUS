from __future__ import annotations

from pathlib import Path

from stand_alone_modules.dedup.scanner import ScanResult


def generate_report(result: ScanResult, output_path: Path) -> None:
    """Write a human-readable markdown report summarising detected duplicates."""
    lines: list[str] = []
    lines.append("# Dedup Scan Report")
    lines.append("")
    lines.append(f"**Dataset path**: `{result.dataset_path}`  ")
    lines.append(f"**Scan timestamp**: {result.scan_timestamp}  ")
    lines.append(f"**Shards scanned**: {result.shards_scanned}  ")
    lines.append(f"**Total duplicate entries found**: {len(result.duplicates)}")
    lines.append("")

    if not result.duplicates:
        lines.append("No duplicates found.")
    else:
        # Group by shard for readability.
        shard_groups: dict[str, list] = {}
        for d in result.duplicates:
            shard_groups.setdefault(d.shard_path, []).append(d)

        total_to_remove = sum(d.duplicates_to_remove for d in result.duplicates)
        shards_affected = len(shard_groups)

        lines.append(f"**Shards with duplicates**: {shards_affected}  ")
        lines.append(f"**Total checkpoint rows to remove**: {total_to_remove}")
        lines.append("")
        lines.append("## Duplicates by Shard")

        for shard_path in sorted(shard_groups):
            entries = shard_groups[shard_path]
            lines.append("")
            lines.append(f"### `{shard_path}`")
            lines.append("")
            lines.append("| Direction Key | Model | Split | Occurrences | To Remove |")
            lines.append("|---|---|---|---|---|")
            for d in sorted(
                entries,
                key=lambda x: (x.direction_key, x.model_name, x.split),
            ):
                lines.append(
                    f"| {d.direction_key} | {d.model_name} | {d.split} "
                    f"| {d.total_occurrences} | {d.duplicates_to_remove} |"
                )

        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Shards with duplicates | {shards_affected} |")
        lines.append(f"| Total duplicate entries | {len(result.duplicates)} |")
        lines.append(f"| Total checkpoint rows to remove | {total_to_remove} |")

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
