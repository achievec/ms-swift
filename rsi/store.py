# Copyright (c) ModelScope Contributors. All rights reserved.
"""SQLite audit log for RSI runs, rounds, steps, checkpoints, and servers."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _dumps(value) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


class Store:

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    scenario TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    current_stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    base_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    promoted_checkpoint TEXT,
                    promoted_kind TEXT,
                    promoted_score REAL,
                    note TEXT,
                    created_at TEXT NOT NULL,
                    closed_at TEXT,
                    UNIQUE(run_id, round_index)
                );
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    result_json TEXT,
                    logs TEXT NOT NULL DEFAULT '',
                    command TEXT NOT NULL DEFAULT '',
                    backend TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    round_index INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS services (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    checkpoint TEXT NOT NULL,
                    served_model_name TEXT NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    api_key TEXT,
                    status TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    logs TEXT NOT NULL DEFAULT '',
                    command TEXT NOT NULL DEFAULT '',
                    started_at TEXT,
                    ended_at TEXT
                );
                CREATE TABLE IF NOT EXISTS fetches (
                    kind TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (kind, repo_id)
                );
                CREATE INDEX IF NOT EXISTS idx_steps_run ON steps(run_id, round_index, id);
                CREATE INDEX IF NOT EXISTS idx_rounds_run ON rounds(run_id, round_index);
                """)
            columns = {row[1] for row in conn.execute('PRAGMA table_info(runs)')}
            if 'model_type' not in columns:
                conn.execute('ALTER TABLE runs ADD COLUMN model_type TEXT')
            if 'template' not in columns:
                conn.execute('ALTER TABLE runs ADD COLUMN template TEXT')

    @contextmanager
    def transaction(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def recover_interrupted(self):
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE steps SET status='failed', "
                "logs = logs || ?, ended_at = COALESCE(ended_at, ?) "
                "WHERE status='running'", ('\n[rsi] 进程在该步骤完成前退出，已标为失败。', now))
            conn.execute("UPDATE services SET status='stopped', ended_at = COALESCE(ended_at, ?) WHERE status='running'",
                         (now, ))

    def create_run(self, run_id, name, scenario, base_model, model_type='', template='') -> None:
        now = utcnow()
        with self.transaction() as conn:
            conn.execute(
                'INSERT INTO runs (id, name, scenario, base_model, model_type, template, round_index, current_stage, '
                'status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)',
                (run_id, name, scenario, base_model, model_type or '', template or '', 'data', 'active', now, now))
            conn.execute(
                'INSERT INTO rounds (run_id, round_index, base_model, status, created_at) VALUES (?, 1, ?, ?, ?)',
                (run_id, base_model, 'active', now))

    def touch(self, conn, run_id):
        conn.execute('UPDATE runs SET updated_at=? WHERE id=?', (utcnow(), run_id))

    def list_runs(self) -> list:
        with self.transaction() as conn:
            rows = conn.execute('SELECT * FROM runs ORDER BY updated_at DESC').fetchall()
        return [dict(row) for row in rows]

    def delete_run(self, run_id) -> bool:
        with self.transaction() as conn:
            row = conn.execute('SELECT id FROM runs WHERE id=?', (run_id, )).fetchone()
            if not row:
                return False
            conn.execute('DELETE FROM services WHERE run_id=?', (run_id, ))
            conn.execute('DELETE FROM checkpoints WHERE run_id=?', (run_id, ))
            conn.execute('DELETE FROM steps WHERE run_id=?', (run_id, ))
            conn.execute('DELETE FROM rounds WHERE run_id=?', (run_id, ))
            conn.execute('DELETE FROM runs WHERE id=?', (run_id, ))
        return True

    def get_run(self, run_id) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute('SELECT * FROM runs WHERE id=?', (run_id, )).fetchone()
        return dict(row) if row else None

    def get_round(self, run_id, round_index) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute('SELECT * FROM rounds WHERE run_id=? AND round_index=?',
                               (run_id, round_index)).fetchone()
        return dict(row) if row else None

    def list_rounds(self, run_id) -> list:
        with self.transaction() as conn:
            rows = conn.execute('SELECT * FROM rounds WHERE run_id=? ORDER BY round_index', (run_id, )).fetchall()
        return [dict(row) for row in rows]

    def list_steps(self, run_id, round_index=None) -> list:
        with self.transaction() as conn:
            if round_index is None:
                rows = conn.execute('SELECT * FROM steps WHERE run_id=? ORDER BY id', (run_id, )).fetchall()
            else:
                rows = conn.execute('SELECT * FROM steps WHERE run_id=? AND round_index=? ORDER BY id',
                                    (run_id, round_index)).fetchall()
        return [dict(row) for row in rows]

    def latest_active_steps(self, run_id, round_index) -> dict:
        active = {}
        for row in self.list_steps(run_id, round_index):
            if row['status'] == 'superseded':
                continue
            active[row['stage']] = row
        return active

    def supersede(self, conn, run_id, round_index, stages: list):
        if not stages:
            return
        marks = ','.join('?' for _ in stages)
        conn.execute(
            f"UPDATE steps SET status='superseded' WHERE run_id=? AND round_index=? "
            f"AND stage IN ({marks}) AND status!='superseded'", [run_id, round_index, *stages])

    def deactivate_checkpoints(self, conn, run_id, round_index, kinds: list):
        if not kinds:
            return
        marks = ','.join('?' for _ in kinds)
        conn.execute(
            f'UPDATE checkpoints SET active=0 WHERE run_id=? AND round_index=? AND kind IN ({marks})',
            [run_id, round_index, *kinds])

    def insert_step(self, conn, run_id, round_index, stage, params, backend) -> int:
        now = utcnow()
        cursor = conn.execute(
            'INSERT INTO steps (run_id, round_index, stage, status, params_json, result_json, logs, command, backend, started_at) '
            "VALUES (?, ?, ?, 'running', ?, NULL, '', '', ?, ?)",
            (run_id, round_index, stage, _dumps(params), backend, now))
        self.touch(conn, run_id)
        return int(cursor.lastrowid)

    def finish_step(self, conn, step_id, status, result, logs, command):
        conn.execute(
            'UPDATE steps SET status=?, result_json=?, logs=?, command=?, ended_at=? WHERE id=?',
            (status, _dumps(result or {}), logs or '', command or '', utcnow(), step_id))

    def add_checkpoint(self, conn, run_id, round_index, kind, path, metrics):
        conn.execute(
            'INSERT INTO checkpoints (run_id, round_index, kind, path, metrics_json, active, created_at) '
            'VALUES (?, ?, ?, ?, ?, 1, ?)', (run_id, round_index, kind, path, _dumps(metrics), utcnow()))

    def active_checkpoint(self, run_id, round_index, kind) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute(
                'SELECT * FROM checkpoints WHERE run_id=? AND round_index=? AND kind=? AND active=1 ORDER BY id DESC LIMIT 1',
                (run_id, round_index, kind)).fetchone()
        return dict(row) if row else None

    def list_checkpoints(self, run_id) -> list:
        with self.transaction() as conn:
            rows = conn.execute('SELECT * FROM checkpoints WHERE run_id=? ORDER BY id', (run_id, )).fetchall()
        return [dict(row) for row in rows]

    def set_current_stage(self, conn, run_id, stage):
        conn.execute('UPDATE runs SET current_stage=?, updated_at=? WHERE id=?', (stage, utcnow(), run_id))

    def set_run_status(self, conn, run_id, status):
        conn.execute('UPDATE runs SET status=?, updated_at=? WHERE id=?', (status, utcnow(), run_id))

    def close_round(self, conn, run_id, round_index, note):
        conn.execute(
            'UPDATE rounds SET status=?, note=?, closed_at=? WHERE run_id=? AND round_index=? AND status=?',
            ('closed', note, utcnow(), run_id, round_index, 'active'))

    def clear_promotion(self, conn, run_id, round_index):
        conn.execute(
            'UPDATE rounds SET promoted_checkpoint=NULL, promoted_kind=NULL, promoted_score=NULL '
            'WHERE run_id=? AND round_index=?', (run_id, round_index))

    def set_promotion(self, conn, run_id, round_index, checkpoint, kind, score):
        conn.execute(
            'UPDATE rounds SET promoted_checkpoint=?, promoted_kind=?, promoted_score=? WHERE run_id=? AND round_index=?',
            (checkpoint, kind, score, run_id, round_index))

    def close_round_and_open_next(self, conn, run_id, round_index, note, next_base):
        now = utcnow()
        conn.execute('UPDATE rounds SET status=?, note=?, closed_at=? WHERE run_id=? AND round_index=? AND status=?',
                     ('closed', note, now, run_id, round_index, 'active'))
        next_index = round_index + 1
        conn.execute(
            'INSERT INTO rounds (run_id, round_index, base_model, status, created_at) VALUES (?, ?, ?, ?, ?)',
            (run_id, next_index, next_base, 'active', now))
        conn.execute('UPDATE runs SET round_index=?, current_stage=?, updated_at=? WHERE id=?',
                     (next_index, 'data', now, run_id))
        return next_index

    def latest_promotion(self, run_id) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute(
                'SELECT * FROM rounds WHERE run_id=? AND promoted_checkpoint IS NOT NULL '
                'ORDER BY round_index DESC LIMIT 1', (run_id, )).fetchone()
        return dict(row) if row else None

    def insert_service(self, conn, payload: dict) -> int:
        cursor = conn.execute(
            'INSERT INTO services (run_id, checkpoint, served_model_name, host, port, api_key, status, backend, '
            'params_json, logs, command, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                payload['run_id'],
                payload['checkpoint'],
                payload['served_model_name'],
                payload['host'],
                payload['port'],
                payload.get('api_key') or '',
                'running',
                payload['backend'],
                _dumps(payload.get('params') or {}),
                payload.get('logs') or '',
                payload.get('command') or '',
                utcnow(),
            ))
        self.touch(conn, payload['run_id'])
        return int(cursor.lastrowid)

    def latest_service(self, run_id) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute('SELECT * FROM services WHERE run_id=? ORDER BY id DESC LIMIT 1', (run_id, )).fetchone()
        return dict(row) if row else None

    def get_service(self, service_id) -> dict | None:
        with self.transaction() as conn:
            row = conn.execute('SELECT * FROM services WHERE id=?', (service_id, )).fetchone()
        return dict(row) if row else None

    def stop_service_row(self, conn, service_id, logs=None):
        if logs is None:
            conn.execute("UPDATE services SET status='stopped', ended_at=? WHERE id=?", (utcnow(), service_id))
        else:
            conn.execute("UPDATE services SET status='stopped', ended_at=?, logs=? WHERE id=?",
                         (utcnow(), logs, service_id))

    def upsert_fetch(self, row: dict) -> None:
        # Update-or-insert. ``INSERT ... ON CONFLICT`` is SQLite 3.24+ and older
        # intranet builds reject it with ``near "ON": syntax error``. The row key
        # is the existing primary key (kind, repo_id).
        kind = row['kind']
        repo_id = row['repo_id']
        local_path = row.get('local_path') or ''
        state = row['state']
        message = row.get('message') or ''
        updated_at = utcnow()
        with self.transaction() as conn:
            found = conn.execute('SELECT 1 FROM fetches WHERE kind=? AND repo_id=?', (kind, repo_id)).fetchone()
            if found:
                conn.execute(
                    'UPDATE fetches SET local_path=?, state=?, message=?, updated_at=? WHERE kind=? AND repo_id=?',
                    (local_path, state, message, updated_at, kind, repo_id))
            else:
                conn.execute(
                    'INSERT INTO fetches (kind, repo_id, local_path, state, message, updated_at) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (kind, repo_id, local_path, state, message, updated_at))

    def list_fetches(self) -> list:
        with self.transaction() as conn:
            rows = conn.execute('SELECT * FROM fetches ORDER BY updated_at DESC').fetchall()
        return [dict(row) for row in rows]

    def list_services(self, run_id) -> list:
        with self.transaction() as conn:
            rows = conn.execute('SELECT * FROM services WHERE run_id=? ORDER BY id', (run_id, )).fetchall()
        return [dict(row) for row in rows]
