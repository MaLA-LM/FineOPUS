#!/bin/bash
# Script to check the status of array tasks

# Allow specifying job name as argument, otherwise use default
JOB_NAME="${1:-fineopus-reLID-glotlid}"
STATUS_LOG="../logs/${JOB_NAME}/status/task_status.log"
LOG_DIR="../logs/${JOB_NAME}"

if [ ! -f "$STATUS_LOG" ]; then
    echo "Status log not found at: $STATUS_LOG"
    echo ""
    echo "Usage: $0 [job_name]"
    echo "Example: $0 fineopus-reLID-glotlid"
    exit 1
fi

echo "=================================================="
echo "Task Status Summary for: ${JOB_NAME}"
echo "=================================================="
echo ""

# Count different statuses (with deduplication for started tasks)
# Count unique task IDs that have started (deduplicate by Task ID)
total_started=$(grep "STARTED" "$STATUS_LOG" 2>/dev/null | grep -oP 'Task \K[0-9]+' | sort -u | wc -l)
total_started=${total_started:-0}

# Get unique successful task IDs
mapfile -t success_task_ids < <(grep "SUCCESS" "$STATUS_LOG" 2>/dev/null | grep -oP 'Task \K[0-9]+' | sort -u)
total_success=${#success_task_ids[@]}

# Get task IDs that have failed (from status log)
mapfile -t failed_task_ids_from_log < <(grep "FAILED" "$STATUS_LOG" 2>/dev/null | grep -oP 'Task \K[0-9]+' | sort -u)

# Check for OOM errors and timeout by scanning .err files
declare -A oom_job_to_task    # job_id -> task_id mapping
declare -A timeout_job_to_task # job_id -> task_id mapping
oom_tasks=()
timeout_tasks=()

if [ -d "$LOG_DIR" ]; then
    shopt -s nullglob  # Make glob patterns return empty if no match
    for err_file in "$LOG_DIR"/*.err; do
        if [ -f "$err_file" ]; then
            # Extract job ID from filename (e.g., jobname_12345.err -> 12345)
            job_id=$(basename "$err_file" | grep -oP '_\K[0-9]+(?=\.err$)')
            
            # Find corresponding task ID from status log
            task_id=$(grep "JobID: $job_id" "$STATUS_LOG" 2>/dev/null | grep -oP 'Task \K[0-9]+' | head -1)
            
            # Check the last 20 lines of each .err file for OOM indicators
            if tail -20 "$err_file" 2>/dev/null | grep -qi "out of memory\|oom"; then
                if [ -n "$job_id" ]; then
                    oom_tasks+=("$job_id")
                    if [ -n "$task_id" ]; then
                        oom_job_to_task["$job_id"]="$task_id"
                    fi
                fi
            # Check for timeout (TIME LIMIT or TIMEOUT)
            elif tail -20 "$err_file" 2>/dev/null | grep -qi "TIME LIMIT\|TIMEOUT\|DUE TO TIME"; then
                if [ -n "$job_id" ]; then
                    timeout_tasks+=("$job_id")
                    if [ -n "$task_id" ]; then
                        timeout_job_to_task["$job_id"]="$task_id"
                    fi
                fi
            fi
        fi
    done
    shopt -u nullglob  # Restore default behavior
fi

# Build list of failed task IDs (OOM + Timeout), excluding those that were later successful
declare -A failed_task_ids_map
for job_id in "${oom_tasks[@]}"; do
    task_id="${oom_job_to_task[$job_id]}"
    if [ -n "$task_id" ]; then
        # Check if this task was later successful
        is_success=false
        for success_id in "${success_task_ids[@]}"; do
            if [ "$task_id" = "$success_id" ]; then
                is_success=true
                break
            fi
        done
        if [ "$is_success" = false ]; then
            failed_task_ids_map["$task_id"]="OOM:$job_id"
        fi
    fi
done

for job_id in "${timeout_tasks[@]}"; do
    task_id="${timeout_job_to_task[$job_id]}"
    if [ -n "$task_id" ]; then
        # Check if this task was later successful or already marked as OOM
        is_success=false
        for success_id in "${success_task_ids[@]}"; do
            if [ "$task_id" = "$success_id" ]; then
                is_success=true
                break
            fi
        done
        # Only add if not successful and not already in failed list (OOM takes priority)
        if [ "$is_success" = false ] && [ -z "${failed_task_ids_map[$task_id]}" ]; then
            failed_task_ids_map["$task_id"]="Timeout:$job_id"
        fi
    fi
done

# Also include tasks marked as FAILED in the log but not in OOM/Timeout
for task_id in "${failed_task_ids_from_log[@]}"; do
    # Check if this task was later successful
    is_success=false
    for success_id in "${success_task_ids[@]}"; do
        if [ "$task_id" = "$success_id" ]; then
            is_success=true
            break
        fi
    done
    # Only add if not successful and not already in failed list
    if [ "$is_success" = false ] && [ -z "${failed_task_ids_map[$task_id]}" ]; then
        # Find the job ID for this task
        job_id=$(grep "FAILED.*Task $task_id " "$STATUS_LOG" 2>/dev/null | grep -oP 'JobID: \K[0-9]+' | tail -1)
        failed_task_ids_map["$task_id"]="Unknown:$job_id"
    fi
done

total_failed=${#failed_task_ids_map[@]}
total_oom=${#oom_tasks[@]}
total_timeout=${#timeout_tasks[@]}

echo "Total tasks started: $total_started"
echo "Successfully completed: $total_success"
echo "Failed: $total_failed"
echo "  - OOM errors: $total_oom"
if [ $total_oom -gt 0 ]; then
    echo "    OOM Job IDs: ${oom_tasks[*]}"
fi
echo "  - Timeout errors: $total_timeout"
if [ $total_timeout -gt 0 ]; then
    echo "    Timeout Job IDs: ${timeout_tasks[*]}"
fi
echo ""

# Display OOM tasks details
if [ $total_oom -gt 0 ]; then
    echo "=================================================="
    echo "OOM Error Tasks:"
    echo "=================================================="
    for job_id in "${oom_tasks[@]}"; do
        # Try to find corresponding task info from status log
        task_info=$(grep "JobID: $job_id" "$STATUS_LOG" 2>/dev/null || echo "")
        if [ -n "$task_info" ]; then
            task_num=$(echo "$task_info" | grep -oP 'Task \K[0-9]+' | head -1)
            filelist=$(echo "$task_info" | grep -oP 'filelist_\K[0-9]+' | head -1)
            echo "  Job $job_id - Task $task_num (filelist_${filelist}.txt) - OOM Error"
        else
            echo "  Job $job_id - OOM Error (no task info found in log)"
        fi
    done
    echo ""
fi

# Display Timeout tasks details
if [ $total_timeout -gt 0 ]; then
    echo "=================================================="
    echo "Timeout Tasks:"
    echo "=================================================="
    for job_id in "${timeout_tasks[@]}"; do
        # Try to find corresponding task info from status log
        task_info=$(grep "JobID: $job_id" "$STATUS_LOG" 2>/dev/null || echo "")
        if [ -n "$task_info" ]; then
            task_num=$(echo "$task_info" | grep -oP 'Task \K[0-9]+' | head -1)
            filelist=$(echo "$task_info" | grep -oP 'filelist_\K[0-9]+' | head -1)
            echo "  Job $job_id - Task $task_num (filelist_${filelist}.txt) - Timeout"
        else
            echo "  Job $job_id - Timeout (no task info found in log)"
        fi
    done
    echo ""
fi
echo ""

if [ "$total_failed" -gt 0 ]; then
    echo "=================================================="
    echo "All Failed Tasks (OOM + Timeout + Other):"
    echo "=================================================="
    # Sort task IDs numerically
    for task_id in $(printf '%s\n' "${!failed_task_ids_map[@]}" | sort -n); do
        info="${failed_task_ids_map[$task_id]}"
        error_type=$(echo "$info" | cut -d':' -f1)
        job_id=$(echo "$info" | cut -d':' -f2)
        
        # Find filelist from status log
        filelist=$(grep "Task $task_id " "$STATUS_LOG" 2>/dev/null | grep -oP 'filelist_\K[0-9]+' | head -1)
        
        if [ "$error_type" = "Unknown" ]; then
            echo "  Task $task_id (filelist_${filelist}.txt) - Job $job_id"
        else
            echo "  Task $task_id (filelist_${filelist}.txt) - Job $job_id - $error_type Error"
        fi
    done
    echo ""
    
    echo "=================================================="
    echo "Tasks to Rerun:"
    echo "=================================================="
    echo "You can rerun individual failed tasks with:"
    echo ""
    for task_id in $(printf '%s\n' "${!failed_task_ids_map[@]}" | sort -n); do
        echo "sbatch --array=$task_id re_lang_identify.sh"
    done
    echo ""
    echo "Or rerun all failed tasks at once:"
    failed_tasks_list=$(printf '%s\n' "${!failed_task_ids_map[@]}" | sort -n | tr '\n' ',' | sed 's/,$//')
    if [ -n "$failed_tasks_list" ]; then
        echo "sbatch --array=$failed_tasks_list re_lang_identify.sh"
    fi
fi

# echo ""
# echo "=================================================="
# echo "Recent Activity (last 5 entries):"
# echo "=================================================="
# tail -5 "$STATUS_LOG"
# echo ""

