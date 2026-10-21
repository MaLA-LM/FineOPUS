#!/usr/bin/env python3
"""Shared SLURM submitter. Unknown options are forwarded to the Python worker."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "/scratch/project_462001509/members/zihao/FineOPUS/FineOPUS-Filtered-Stage6"
STAGES = {
    "collect": ("thresholds/collect_data_stats.sh", True, 4, "32G", "2-00:00:00"),
    "thresholds": ("thresholds/compute_thresholds.sh", False, 1, "16G", "0-02:00:00"),
    "filter": ("filter/apply_thresholds.sh", True, 4, "32G", "2-00:00:00"),
    "merge": ("filter/merge_filter_stats.sh", False, 1, "8G", "0-02:00:00"),
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="For filter/merge, --out-dir defaults to {}.".format(DEFAULT_OUTPUT))
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--tasks", type=int, help="Array size: default 64 for collect/filter; thresholds/merge require 1")
    p.add_argument("--account", default="project_462001249",
                   help="SLURM account for every stage (default: project_462001249)")
    p.add_argument("--partition", default="small")
    p.add_argument("--cpus", type=int, help="CPUs per task: default 4 for arrays, 1 for single jobs")
    p.add_argument("--mem", help="Default: collect/filter 32G, thresholds 16G, merge 8G")
    p.add_argument("--time", help="Default: 2 days for arrays, 2 hours for single jobs")
    p.add_argument("--dependency", help="For example afterok:123456")
    default_python = HERE.parent / ".venv/bin/python"
    p.add_argument("--python", default=str(default_python) if default_python.exists() else "python3")
    p.add_argument("--dry-run", action="store_true")
    args, worker_args = p.parse_known_args()
    worker_name, is_array, cpus, memory, time_limit = STAGES[args.stage]
    args.tasks = args.tasks if args.tasks is not None else (64 if is_array else 1)
    args.cpus = args.cpus if args.cpus is not None else cpus
    args.mem = args.mem or memory
    args.time = args.time or time_limit
    if min(args.tasks, args.cpus) < 1:
        p.error("tasks and cpus must be positive")
    if not is_array and args.tasks != 1:
        p.error("thresholds and merge run as single jobs; --tasks must be 1")
    log_dir = HERE.parent / "logs/joint_filter"
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
    worker = HERE / worker_name
    command = ["sbatch", "--parsable", f"--job-name=joint_{args.stage}", f"--account={args.account}",
               f"--partition={args.partition}", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}",
               f"--output={log_dir}/%x_%A_%a.out", f"--error={log_dir}/%x_%A_%a.err"]
    if is_array:
        command.append(f"--array=0-{args.tasks - 1}")
    if args.dependency:
        command.append(f"--dependency={args.dependency}")
    # Accept both wrapper-style --scored-dir and Python-style --scored_dir.
    forwarded = []
    for arg in worker_args:
        if arg.startswith("--"):
            key, separator, value = arg.partition("=")
            arg = "--" + key[2:].replace("-", "_") + separator + value
        forwarded.append(arg)
    worker_args = forwarded
    if args.stage in ("filter", "merge") and not any(
            arg == "--out_dir" or arg.startswith("--out_dir=") for arg in worker_args):
        worker_args += ["--out_dir", DEFAULT_OUTPUT]
    # Slurm copies the shell worker into its spool directory. BASH_SOURCE in
    # that worker no longer points into this repository, so pass the Python
    # entrypoint explicitly instead of resolving it relative to the worker.
    command += [str(worker), args.python, str(worker.with_suffix(".py")), *worker_args]
    # Login-node python3 may predate shlex.join (Python 3.8).
    # Keep stdout to sbatch's parsable job ID on submission, so callers can use
    # JOB=$(bash submit_*.sh ...) and --dependency "afterok:${JOB%%;*}".
    print(" ".join(shlex.quote(part) for part in command),
          file=sys.stdout if args.dry_run else sys.stderr, flush=True)
    if not args.dry_run:
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
        except FileNotFoundError:
            print("sbatch was not found; submit from a SLURM login node.", file=sys.stderr)
            sys.exit(127)
        if result.stderr:
            sys.stderr.write(result.stderr)
        if result.returncode:
            print("Submission failed (account={}, partition={}); no job ID returned."
                  .format(args.account, args.partition), file=sys.stderr)
            sys.exit(result.returncode)
        sys.stdout.write(result.stdout)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
