"""Single-writer SQLite registry for deterministic resumable studies."""

from __future__ import annotations

import fcntl
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lolpredictor.optimization.schedule import TrialSpec

STUDY_REGISTRY_SCHEMA_VERSION = "1"
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password)(\s*[:=]\s*)[^\s,;]+"
)


def _now_text() -> str:
    return datetime.now(UTC).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sanitize_trial_error(error: BaseException) -> str:
    """Keep a bounded diagnostic without persisting credential-like values."""
    message = f"{type(error).__name__}: {error}"
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1\2<redacted>", message)
    return sanitized[:4000]


class StudyRegistry:
    """Persistent trial state with writes owned by one orchestrator process."""

    def __init__(self, study_directory: Path) -> None:
        self.directory = study_directory.resolve()
        self.database_path = self.directory / "study.sqlite3"
        self.lock_path = self.directory / ".study.lock"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> StudyRegistry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS study_state (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY,
                    outer_fold INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    feature_mode TEXT NOT NULL,
                    spec_hash TEXT NOT NULL UNIQUE,
                    spec_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending', 'running', 'completed', 'failed')
                    ),
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT,
                    completed_at TEXT,
                    duration_seconds REAL,
                    worker_pid INTEGER,
                    result_json TEXT,
                    error_text TEXT
                )
                """
            )

    @contextmanager
    def exclusive_lock(self) -> Iterator[None]:
        """Prevent two orchestrators from mutating one study at the same time."""
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"Another optimizer process owns the study lock: {self.lock_path}"
                ) from error
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def initialize(
        self,
        *,
        contract: dict[str, Any],
        specs: list[TrialSpec],
    ) -> None:
        """Create a study or verify that an existing study has the exact contract."""
        expected_metadata = {
            "registry_schema_version": STUDY_REGISTRY_SCHEMA_VERSION,
            **contract,
        }
        rows = self._connection.execute(
            "SELECT key, value_json FROM study_metadata ORDER BY key"
        ).fetchall()
        existing = {str(row["key"]): json.loads(row["value_json"]) for row in rows}
        if existing:
            if existing != expected_metadata:
                mismatched = sorted(
                    key
                    for key in set(existing) | set(expected_metadata)
                    if existing.get(key) != expected_metadata.get(key)
                )
                raise ValueError("Study registry contract mismatch for: " + ", ".join(mismatched))
            self._validate_existing_specs(specs)
            return

        created_at = _now_text()
        with self._connection:
            self._connection.executemany(
                "INSERT INTO study_metadata (key, value_json) VALUES (?, ?)",
                [(key, _json_text(value)) for key, value in sorted(expected_metadata.items())],
            )
            self._connection.executemany(
                """
                INSERT INTO trials (
                    trial_id,
                    outer_fold,
                    family,
                    feature_mode,
                    spec_hash,
                    spec_json,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                [
                    (
                        spec.trial_id,
                        spec.outer_fold,
                        spec.family,
                        spec.feature_mode,
                        spec.spec_hash,
                        _json_text(spec.to_dict()),
                    )
                    for spec in specs
                ],
            )
            self._connection.executemany(
                "INSERT INTO study_state (key, value_json) VALUES (?, ?)",
                [
                    ("created_at", _json_text(created_at)),
                    ("active_elapsed_seconds", _json_text(0.0)),
                    ("last_updated_at", _json_text(created_at)),
                ],
            )

    def _validate_existing_specs(self, specs: list[TrialSpec]) -> None:
        rows = self._connection.execute(
            "SELECT trial_id, spec_json FROM trials ORDER BY trial_id"
        ).fetchall()
        existing = [(int(row["trial_id"]), str(row["spec_json"])) for row in rows]
        expected = [
            (spec.trial_id, _json_text(spec.to_dict()))
            for spec in sorted(specs, key=lambda item: item.trial_id)
        ]
        if existing != expected:
            raise ValueError("Study trial schedule does not match the existing registry")

    def metadata(self) -> dict[str, Any]:
        rows = self._connection.execute(
            "SELECT key, value_json FROM study_metadata ORDER BY key"
        ).fetchall()
        return {str(row["key"]): json.loads(row["value_json"]) for row in rows}

    def state(self) -> dict[str, Any]:
        rows = self._connection.execute(
            "SELECT key, value_json FROM study_state ORDER BY key"
        ).fetchall()
        return {str(row["key"]): json.loads(row["value_json"]) for row in rows}

    def _set_state(self, key: str, value: Any) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO study_state (key, value_json) VALUES (?, ?)",
            (key, _json_text(value)),
        )

    def add_active_elapsed(self, seconds: float) -> None:
        state = self.state()
        elapsed = float(state.get("active_elapsed_seconds", 0.0)) + max(0.0, seconds)
        with self._connection:
            self._set_state("active_elapsed_seconds", elapsed)
            self._set_state("last_updated_at", _now_text())

    def ensure_state_value(self, key: str, value: Any) -> None:
        """Set an immutable derived-study value or verify its prior value."""
        state = self.state()
        if key in state:
            if state[key] != value:
                raise ValueError(f"Study state mismatch for immutable value: {key}")
            return
        with self._connection:
            self._set_state(key, value)
            self._set_state("last_updated_at", _now_text())

    def reset_interrupted_trials(self) -> int:
        """Return orphaned running trials to pending during a locked resume."""
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE trials
                SET
                    status = 'pending',
                    started_at = NULL,
                    worker_pid = NULL,
                    duration_seconds = NULL,
                    completed_at = NULL,
                    result_json = NULL,
                    error_text = NULL
                WHERE status = 'running'
                """
            )
        return int(cursor.rowcount)

    def pending_specs(self) -> list[TrialSpec]:
        rows = self._connection.execute(
            "SELECT spec_json FROM trials WHERE status = 'pending' ORDER BY trial_id"
        ).fetchall()
        return [TrialSpec.from_dict(json.loads(row["spec_json"])) for row in rows]

    def claim(self, trial_id: int) -> None:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE trials
                SET
                    status = 'running',
                    attempt_count = attempt_count + 1,
                    started_at = ?,
                    completed_at = NULL,
                    duration_seconds = NULL,
                    worker_pid = NULL,
                    result_json = NULL,
                    error_text = NULL
                WHERE trial_id = ? AND status = 'pending'
                """,
                (_now_text(), trial_id),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Trial {trial_id} was not pending when claimed")

    def complete(
        self,
        trial_id: int,
        *,
        result: dict[str, Any],
        duration_seconds: float,
        worker_pid: int,
    ) -> None:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE trials
                SET
                    status = 'completed',
                    completed_at = ?,
                    duration_seconds = ?,
                    worker_pid = ?,
                    result_json = ?,
                    error_text = NULL
                WHERE trial_id = ? AND status = 'running'
                """,
                (
                    _now_text(),
                    duration_seconds,
                    worker_pid,
                    _json_text(result),
                    trial_id,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Trial {trial_id} was not running when completed")

    def fail(
        self,
        trial_id: int,
        *,
        error: BaseException,
        duration_seconds: float,
        worker_pid: int | None,
    ) -> None:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE trials
                SET
                    status = 'failed',
                    completed_at = ?,
                    duration_seconds = ?,
                    worker_pid = ?,
                    result_json = NULL,
                    error_text = ?
                WHERE trial_id = ? AND status = 'running'
                """,
                (
                    _now_text(),
                    duration_seconds,
                    worker_pid,
                    sanitize_trial_error(error),
                    trial_id,
                ),
            )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Trial {trial_id} was not running when failed")

    def status_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT status, COUNT(*) AS count FROM trials GROUP BY status"
        ).fetchall()
        values = {status: 0 for status in ("pending", "running", "completed", "failed")}
        values.update({str(row["status"]): int(row["count"]) for row in rows})
        return values

    def trial_records(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT
                trial_id,
                outer_fold,
                family,
                feature_mode,
                spec_hash,
                spec_json,
                status,
                attempt_count,
                started_at,
                completed_at,
                duration_seconds,
                worker_pid,
                result_json,
                error_text
            FROM trials
            ORDER BY trial_id
            """
        ).fetchall()
        return [
            {
                "trial_id": int(row["trial_id"]),
                "outer_fold": int(row["outer_fold"]),
                "family": str(row["family"]),
                "feature_mode": str(row["feature_mode"]),
                "spec_hash": str(row["spec_hash"]),
                "spec": json.loads(row["spec_json"]),
                "status": str(row["status"]),
                "attempt_count": int(row["attempt_count"]),
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "duration_seconds": row["duration_seconds"],
                "worker_pid": row["worker_pid"],
                "result": (
                    json.loads(row["result_json"]) if row["result_json"] is not None else None
                ),
                "error": row["error_text"],
            }
            for row in rows
        ]
