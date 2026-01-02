# Exact Deduplication Module

This module performs exact deduplication on parallel corpus data across language pairs. It removes duplicate sentence pairs based on the `source_text` and `target_text` columns and shards the output for efficient processing.

## Overview

The deduplication process:
- **Identifies exact duplicates** within each language pair directory
- **Removes duplicate rows** based on source and target text columns
- **Shards output** into manageable chunks (default: 100M lines per shard)
- **Tracks statistics** including input/output row counts and removal rates
- **Manages memory efficiently** using DuckDB's disk-based spilling for large datasets

## Files

### Core Scripts

- **`exact_dedup.py`** - Main Python script for deduplication
  - Uses DuckDB for efficient SQL-based deduplication
  - Handles disk-based memory management to prevent OOM on large datasets
  - Partitions output by shard ID for distributed processing
  - Flattens Hive partitions into individual parquet files

- **`dedup_exact_all.sh`** - Orchestration script for batch processing
  - Discovers all incomplete language pair folders
  - Creates a task list of folders needing processing
  - Submits Slurm array jobs for parallel deduplication
  - Tracks progress using the stats CSV file

- **`exact_dedup_chunk.slurm`** - Slurm job template
  - Processes multiple language pairs per array task
  - Manages resource allocation (CPUs, memory, time)
  - Logs execution details for monitoring

### Output Files

- **`exact_dedup_stats.csv`** - Cumulative statistics log
  - Tracks deduplication results for each language pair
  - Columns: `lang_pair`, `input_rows`, `output_rows`, `rows_removed`, `timestamp`
  - Used to identify incomplete folders and avoid reprocessing

## Usage

### Basic Usage (Single Language Pair)

```bash
python exact_dedup.py \
  --data_dir /path/to/lang_pair \
  --out_dir /path/to/output \
  --temp_dir /path/to/temp_files \
  --stats_file /path/to/stats.csv \
  --src_col "source_text" \
  --trg_col "target_text"
```

### Batch Processing (Multiple Language Pairs)

1. **Configure paths** in `dedup_exact_all.sh`:
   ```bash
   DATA_DIR="/path/to/input/data"      # Root directory with language pair folders
   OUT_DIR="/path/to/output"            # Destination for deduplicated data
   STATS_FILE="/path/to/stats.csv"      # Statistics tracking file
   TMP_DIR="/path/to/temp"              # Temporary directory for DuckDB
   CHUNK_SIZE=5                         # Folders per Slurm task
   ```

2. **Submit the batch job**:
   ```bash
   bash dedup_exact_all.sh
   ```

   The script will:
   - Scan all folders in `DATA_DIR`
   - Compare against existing entries in `STATS_FILE`
   - Submit only incomplete folders for processing
   - Create parallel jobs based on `CHUNK_SIZE`

## Parameters

### Python Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Root directory containing input parquet files |
| `--out_dir` | *required* | Root directory for deduplicated output |
| `--stats_file` | *required* | CSV file path for statistics tracking |
| `--temp_dir` | *required* | Directory for DuckDB temporary files (needed for large data) |
| `--src_col` | `source_text` | Column name for source language text |
| `--trg_col` | `target_text` | Column name for target language text |
| `--max_lines` | `100_000_000` | Maximum lines per output shard |
| `--compression` | `SNAPPY` | Compression codec (SNAPPY or ZSTD) |
| `--memory_limit` | `10GB` | RAM limit for DuckDB (e.g., 20GB, 100GB) |

### Slurm Configuration

Edit `exact_dedup_chunk.slurm` to adjust:
- `--cpus-per-task` - CPU cores (default: 16)
- `--mem` - Total memory (default: 224G)
- `--time` - Maximum job duration (default: 72 hours)
- `--partition` - Queue name (default: small)
- `--account` - Project ID for billing

## How It Works

### Deduplication Algorithm

1. **Load all parquet files** from the input directory recursively
2. **Filter null values** - Remove rows where source or target text is NULL
3. **Select distinct rows** - SQL `SELECT DISTINCT` on source and target columns
4. **Add row numbers** - Enumerate deduplicated rows
5. **Calculate shard IDs** - Distribute rows into shards based on `max_lines`
6. **Write partitioned output** - Save as parquet files with Hive partitioning
7. **Flatten partitions** - Reorganize from `shard_id=X/data.parquet` to `{name}_shard_X.parquet`

### Memory Management

- **Disk-based spilling**: Uses physical database file + temp directory instead of in-memory database
- **Configurable memory limit**: Set via `--memory_limit` (e.g., 10GB, 100GB)
- **PRAGMA settings**:
  - `memory_limit` - Hard limit on RAM usage
  - `temp_directory` - Location for intermediate query results
  - `preserve_insertion_order=FALSE` - Disables ordering for speed
  - `threads=4` - Multi-threaded query execution

### Progress Tracking

The `dedup_exact_all.sh` script enables **incremental processing**:
- Checks existing entries in `exact_dedup_stats.csv`
- Only submits folders not yet in the stats file
- Allows rerunning the script safely without reprocessing completed folders
- Creates a temporary `incomplete_folders_dedup.tmp.txt` file during execution

## Output

### Directory Structure

```
/out_dir/{lang_pair}/
  ├── {lang_pair}_shard_0.parquet
  ├── {lang_pair}_shard_1.parquet
  ├── {lang_pair}_shard_2.parquet
  └── ...
```

### Statistics Format

The `exact_dedup_stats.csv` file contains:
```
lang_pair,input_rows,output_rows,rows_removed,timestamp
eng_Latn-fra_Latn,10000000,8500000,1500000,2025-11-30 10:32:24
```

## Performance Considerations

- **Memory**: Allocate sufficient temp space (2-3x the input data size recommended)
- **CPU**: Deduplication is CPU-intensive; use 16+ cores for efficiency
- **Chunk size**: Reduce `CHUNK_SIZE` if jobs timeout (more jobs, lower peak memory per job)


## Dependencies

- Python 3.8+
- `duckdb` - SQL-based deduplication engine

Install dependencies:
```bash
pip install duckdb
```

## References

- DuckDB Documentation: https://duckdb.org/docs/
- Parquet Format: https://parquet.apache.org/
