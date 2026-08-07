"""Persistence for computed verdicts.

A verdict costs a few seconds of GPU time to produce, and the report endpoint
needs it back on a later request from a client that only kept the id. SQLite
keeps that a single file next to the corpus rather than a service to run.

The whole `VerdictResponse` is stored as its serialised JSON in one column.
Normalising it into tables would mean a second schema to keep in step with
`models.py`, and every consumer here wants the entire object anyway — the
report quotes passages, tiers, coverage signals and per-verdict limitations
alike. `claim`, `verdict` and `created_at` are lifted into their own columns
only because they are worth querying without parsing every payload.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.config import settings
from app.models import VerdictResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    claim           TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    payload         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_verifications_created_at
    ON verifications (created_at DESC);
"""


class StoredPayloadError(RuntimeError):
    """A stored row could not be read back as a VerdictResponse.

    Raised instead of letting pydantic's error surface directly, because the
    realistic cause is a schema change since the row was written, and that is
    worth saying plainly rather than as a validation dump.
    """


@dataclass(frozen=True)
class StoredVerification:
    """A verdict as recovered from the store, with when it was computed.

    The timestamp is the store's, not the verdict's: `VerdictResponse` carries
    no date of its own, and a report needs to state when the claim was actually
    assessed rather than when someone later exported it.
    """

    response: VerdictResponse
    created_at: datetime


class VerificationStore:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.verifications_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A connection per operation: FastAPI runs sync endpoints in a thread
        # pool, and sqlite3 connections are not safe to share across threads.
        # At this request volume the open cost is irrelevant.
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save(self, response: VerdictResponse) -> None:
        """Persist a verdict. Never raises — a failed write must not fail /verify.

        The verdict has already been computed and is about to be returned to
        the caller. Losing the ability to export it later is a far smaller loss
        than turning a successful verification into a 500, so a storage failure
        is reported to the log and swallowed.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO verifications "
                    "(verification_id, claim, verdict, created_at, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        response.verification_id,
                        response.claim,
                        response.verdict.value,
                        datetime.now(timezone.utc).isoformat(),
                        response.model_dump_json(),
                    ),
                )
        except sqlite3.Error as exc:
            print(
                f"WARNING: could not persist verification "
                f"{response.verification_id}: {exc}",
                file=sys.stderr,
            )

    def get(self, verification_id: str) -> StoredVerification | None:
        """Fetch a stored verdict, or None if that id was never recorded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM verifications WHERE verification_id = ?",
                (verification_id,),
            ).fetchone()

        if row is None:
            return None

        payload, created_at = row
        try:
            response = VerdictResponse.model_validate(json.loads(payload))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise StoredPayloadError(
                f"Verification {verification_id} is stored but no longer readable "
                f"as a VerdictResponse, which usually means the response schema "
                f"changed after it was written. Re-run the claim to get a fresh "
                f"result. Underlying error: {exc}"
            ) from exc

        return StoredVerification(
            response=response, created_at=datetime.fromisoformat(created_at)
        )


_store: VerificationStore | None = None


def get_store() -> VerificationStore:
    global _store
    if _store is None:
        _store = VerificationStore()
    return _store
