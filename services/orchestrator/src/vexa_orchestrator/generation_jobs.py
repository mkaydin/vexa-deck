"""Durable local track jobs. No database access occurs in the audio callback."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class GenerationJobs:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS batches (
                    id TEXT PRIMARY KEY, theme TEXT NOT NULL, snapshot TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
                    updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS tracks (
                    batch_id TEXT NOT NULL REFERENCES batches(id), slot INTEGER NOT NULL,
                    state TEXT NOT NULL, brief TEXT, asset_path TEXT, asset_id TEXT,
                    plan_attempts INTEGER NOT NULL DEFAULT 0,
                    render_attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT, updated REAL NOT NULL,
                    PRIMARY KEY(batch_id, slot));
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY, batch_id TEXT NOT NULL, slot INTEGER NOT NULL,
                    stage TEXT NOT NULL, outcome TEXT NOT NULL, error TEXT, at REAL NOT NULL);
            """)
            if "revision" not in {row["name"] for row in db.execute("PRAGMA table_info(tracks)")}:
                db.execute("ALTER TABLE tracks ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
            if "planning_revision" not in {
                row["name"] for row in db.execute("PRAGMA table_info(tracks)")
            }:
                db.execute(
                    "ALTER TABLE tracks ADD COLUMN planning_revision INTEGER NOT NULL DEFAULT 0"
                )
            if "evidence" not in {row["name"] for row in db.execute("PRAGMA table_info(attempts)")}:
                db.execute("ALTER TABLE attempts ADD COLUMN evidence TEXT")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def exclusive(self, cancel):
        import fcntl

        with self.path.with_suffix(".lock").open("a") as handle:
            while True:
                if cancel.is_set():
                    raise RuntimeError("generation queue cancelled")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    cancel.wait(0.1)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def batch(self, batch_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if row is None:
            raise ValueError("unknown generation batch")
        return {**dict(row), "snapshot": json.loads(row["snapshot"])}

    def batches(self, limit=20) -> list[dict]:
        with self.connect() as db:
            return [
                {**dict(row), "snapshot": json.loads(row["snapshot"])}
                for row in db.execute(
                    "SELECT * FROM batches ORDER BY created DESC LIMIT ?", (limit,)
                )
            ]

    def tracks(self, batch_id: str) -> list[dict]:
        with self.connect() as db:
            return [
                {**dict(row), "brief": json.loads(row["brief"]) if row["brief"] else None}
                for row in db.execute(
                    "SELECT * FROM tracks WHERE batch_id=? ORDER BY slot", (batch_id,)
                )
            ]

    def open_batch(self, theme: str, snapshot: dict, count: int, *, force_new=False) -> str:
        fingerprint = hashlib.sha256(
            json.dumps(
                {"theme": theme, "snapshot": snapshot, "count": count}, sort_keys=True
            ).encode()
        ).hexdigest()
        now = time.time()
        with self.connect() as db:
            row = (
                None
                if force_new
                else db.execute(
                    "SELECT id FROM batches WHERE fingerprint=? "
                    "AND state IN ('running','interrupted') "
                    "ORDER BY created DESC LIMIT 1",
                    (fingerprint,),
                ).fetchone()
            )
            if row:
                return row["id"]
            batch_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO batches VALUES (?,?,?,?,?,?,?)",
                (batch_id, theme, json.dumps(snapshot), fingerprint, "running", now, now),
            )
            db.executemany(
                "INSERT INTO tracks(batch_id,slot,state,updated) VALUES (?,?,'queued',?)",
                [(batch_id, slot, now) for slot in range(count)],
            )
        return batch_id

    def update(self, batch_id: str, slot: int, **fields):
        allowed = {
            "state",
            "brief",
            "asset_path",
            "asset_id",
            "plan_attempts",
            "render_attempts",
            "error",
            "revision",
        }
        if not fields or not set(fields) <= allowed:
            raise ValueError("invalid track job update")
        if "brief" in fields and fields["brief"] is not None:
            fields["brief"] = json.dumps(fields["brief"])
        fields["updated"] = time.time()
        with self.connect() as db:
            db.execute(
                "UPDATE tracks SET "
                + ",".join(f"{key}=?" for key in fields)
                + " WHERE batch_id=? AND slot=?",
                (*fields.values(), batch_id, slot),
            )

    def mark_batch(self, batch_id: str, state: str):
        with self.connect() as db:
            db.execute(
                "UPDATE batches SET state=?,updated=? WHERE id=?", (state, time.time(), batch_id)
            )

    def attempt(
        self, batch_id: str, slot: int, stage: str, outcome: str, error=None, *, evidence=None
    ):
        with self.connect() as db:
            db.execute(
                "INSERT INTO attempts(batch_id,slot,stage,outcome,error,at,evidence) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    batch_id,
                    slot,
                    stage,
                    outcome,
                    error,
                    time.time(),
                    json.dumps(evidence) if evidence else None,
                ),
            )

    def failure_count(self, batch_id: str) -> int:
        with self.connect() as db:
            return db.execute(
                "SELECT count(*) FROM attempts WHERE batch_id=? AND outcome='failed'", (batch_id,)
            ).fetchone()[0]

    def reset_failed(self, batch_id: str):
        with self.connect() as db:
            db.execute(
                "UPDATE tracks SET state=CASE WHEN brief IS NULL THEN 'queued' "
                "ELSE 'plan_ready' END,plan_attempts=0,render_attempts=0,error=NULL,"
                "asset_path=NULL,asset_id=NULL,revision=revision+1,updated=? WHERE batch_id=? "
                "AND state IN ('rejected','unavailable')",
                (time.time(), batch_id),
            )
        self.mark_batch(batch_id, "interrupted")

    def reset_slot(self, batch_id: str, slot: int, *, replace=False):
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM tracks WHERE batch_id=? AND slot=?", (batch_id, slot)
            ).fetchone()
            if row is None:
                raise ValueError("unknown track slot")
            if not replace and row["state"] not in {"rejected", "unavailable"}:
                raise ValueError("only a rejected or unavailable slot can be retried")
            brief = None if replace else row["brief"]
            db.execute(
                "UPDATE tracks SET state=?,brief=?,plan_attempts=0,render_attempts=0,"
                "error=NULL,asset_path=NULL,asset_id=NULL,revision=revision+1,updated=? "
                ",planning_revision=planning_revision+? "
                "WHERE batch_id=? AND slot=?",
                (
                    "queued" if brief is None else "plan_ready",
                    brief,
                    time.time(),
                    int(replace),
                    batch_id,
                    slot,
                ),
            )
        self.mark_batch(batch_id, "interrupted")

    def attempts(self, batch_id: str, slot: int) -> list[dict]:
        with self.connect() as db:
            return [
                {**dict(row), "evidence": json.loads(row["evidence"]) if row["evidence"] else None}
                for row in db.execute(
                    "SELECT stage,outcome,error,at,evidence FROM attempts "
                    "WHERE batch_id=? AND slot=? ORDER BY id",
                    (batch_id, slot),
                )
            ]
