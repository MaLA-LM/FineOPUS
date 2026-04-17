-- Canonical schema for the OPUS queue. Load this on first connection
-- from queue_db.initialize(). Keep CREATE statements idempotent so
-- schema.sql doubles as a migration-safe "ensure tables" script.

CREATE TABLE IF NOT EXISTS directions (
    direction_key TEXT NOT NULL,
    model TEXT NOT NULL,
    src_lang TEXT NOT NULL,
    tgt_lang TEXT NOT NULL,
    n_sentences INTEGER NOT NULL,
    shard_size INTEGER NOT NULL,
    est_hours REAL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (direction_key, model)
);

CREATE TABLE IF NOT EXISTS jobs (
    direction_key TEXT NOT NULL,
    model TEXT NOT NULL,
    shard_id INTEGER NOT NULL,
    start_idx INTEGER NOT NULL,
    end_idx INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    worker_id TEXT,
    started_at INTEGER,
    finished_at INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    out_path TEXT,
    PRIMARY KEY (direction_key, model, shard_id),
    FOREIGN KEY (direction_key, model) REFERENCES directions(direction_key, model)
);

CREATE INDEX IF NOT EXISTS ix_jobs_model_status ON jobs(model, status);
CREATE INDEX IF NOT EXISTS ix_jobs_status_started ON jobs(status, started_at);

CREATE TABLE IF NOT EXISTS run_events (
    ts INTEGER NOT NULL,
    worker_id TEXT,
    event TEXT NOT NULL,
    direction_key TEXT,
    model TEXT,
    shard_id INTEGER,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS ix_run_events_ts ON run_events(ts);
