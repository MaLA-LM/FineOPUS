from dedup.cli import parse_args
from dedup.deleter import apply_plan
from dedup.report import generate_report
from dedup.scanner import scan_dataset, scan_result_to_plan, write_plan
from utils.logger import logger


def main() -> None:
    args = parse_args()

    if args.command == "scan":
        result = scan_dataset(args.dataset_path)

        plan = scan_result_to_plan(result)
        plan_path = args.output / "dedup_plan.json"
        write_plan(plan, plan_path)
        logger.info("Plan written to: %s", plan_path)

        report_path = args.output / "dedup_report.md"
        generate_report(result, report_path)
        logger.info("Report written to: %s", report_path)

        logger.info(
            "Scan complete: shards_scanned=%d duplicates_found=%d",
            result.shards_scanned,
            len(result.duplicates),
        )

    elif args.command == "apply":
        result = apply_plan(args.plan)
        logger.info(
            "Apply complete: shards_processed=%d checkpoint_lines_removed=%d "
            "part_files_modified=%d part_rows_removed=%d",
            result.shards_processed,
            result.total_checkpoint_lines_removed,
            result.total_part_files_modified,
            result.total_part_rows_removed,
        )


if __name__ == "__main__":
    main()
