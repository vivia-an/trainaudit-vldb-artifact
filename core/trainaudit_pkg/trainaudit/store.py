"""DuckDB-backed append-only trace store.

Two execution modes:

  - **sync** (default): `emit()` does json.dumps + appends to in-memory
    buffer; `_flush_locked()` does executemany at buffer_limit. Predictable
    and ordering-deterministic. Used by tests and CPU smoke runs.

  - **async** (opt-in via `async_mode=True`): `emit()` increments the
    counter and pushes the raw payload (un-serialised) onto a thread-safe
    queue, returning the event_id immediately. A background daemon thread
    drains the queue, serialises payloads, and writes to DuckDB in batches.
    This pushes the json.dumps + INSERT cost off the training step's
    critical path. Used in production to hit paper §4.3 <5% overhead.

The two modes are wire-compatible: same events table schema, same event_id
ordering, same `flush()` semantics for reads. Production code calls
`enable(..., async_mode=True)`; rule code can ignore the difference because
`flush()` waits for the queue to drain before any read.
"""
import atexit
import json
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

import duckdb

from .schema import SCHEMA_VERSION

# Internal type for an in-flight async event before json.dumps. Carrying
# the raw `payload` dict rather than the serialised string is what shifts
# the cost off the critical path.
_PendingEvent = Tuple[int, int, int, str, int, Dict[str, Any]]


class TraceStore:
    """One DuckDB file, one events table, JSON payload.

    Schema is intentionally minimal. Rules read it via SQL or .query().
    """

    def __init__(self, db_path: str = ":memory:", *,
                 async_mode: bool = False,
                 async_queue_size: int = 8192,
                 async_drain_interval_s: float = 0.2):
        self.db_path = db_path
        self.conn = duckdb.connect(str(db_path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id        BIGINT  PRIMARY KEY,
                step            INTEGER,
                rank            INTEGER,
                hookpoint       VARCHAR,
                ts_ns           BIGINT,
                schema_version  INTEGER,
                payload         VARCHAR
            )
        """)
        self._counter = 0
        self._lock = threading.Lock()
        self._buffer = []
        self._buffer_limit = 256
        self._step = 0
        self._rank = 0
        self._closed = False

        # Async-mode state. All `_async_*` attributes only meaningful when
        # async_mode=True.
        self._async_mode = async_mode
        self._async_q: Optional[queue.Queue] = None
        self._async_thread: Optional[threading.Thread] = None
        self._async_done: Optional[threading.Event] = None
        self._async_drain_interval = async_drain_interval_s
        # health counters — readable via `health()`
        self._async_dropped = 0
        self._async_serialize_failures = 0
        self._async_db_failures = 0
        if async_mode:
            self._async_q = queue.Queue(maxsize=async_queue_size)
            self._async_done = threading.Event()
            self._async_thread = threading.Thread(
                target=self._async_drain_loop, daemon=True,
                name="trainaudit-store-drain")
            self._async_thread.start()
            atexit.register(self._async_shutdown)

    # ---- public API ----

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def set_rank(self, rank: int) -> None:
        self._rank = int(rank)

    def emit(self, hookpoint: str, payload: Dict[str, Any], *,
             step: Optional[int] = None, rank: Optional[int] = None) -> int:
        """Append one event. Returns event_id (or -1 if store is closed).

        Sync mode: serialises payload + buffers; flushes at buffer_limit.
        Async mode: pushes raw payload onto queue; returns immediately.
        Counter is monotonically incremented under lock in either mode so
        event_id ordering matches emit-call ordering across threads."""
        if self._closed:
            return -1

        with self._lock:
            self._counter += 1
            event_id = self._counter
        ts = time.time_ns()
        cur_step = self._step if step is None else int(step)
        cur_rank = self._rank if rank is None else int(rank)

        if self._async_mode:
            pending: _PendingEvent = (event_id, cur_step, cur_rank,
                                       hookpoint, ts, payload)
            try:
                self._async_q.put_nowait(pending)
            except queue.Full:
                # Back-pressure: try briefly blocking before dropping.
                # Dropping silently corrupts trace, so retry once.
                try:
                    self._async_q.put(pending, timeout=0.05)
                except queue.Full:
                    self._async_dropped += 1
            return event_id

        # Sync path: serialise here.
        with self._lock:
            self._buffer.append((
                event_id, cur_step, cur_rank, hookpoint, ts,
                SCHEMA_VERSION,
                json.dumps(payload, default=_json_default),
            ))
            if len(self._buffer) >= self._buffer_limit:
                self._flush_locked()
        return event_id

    def flush(self) -> None:
        """Wait for all pending events (async queue + sync buffer) to land
        in DuckDB. Safe for rule code to call before reading."""
        if self._async_mode and self._async_q is not None:
            # Wait until bg thread drained queue (best-effort with timeout).
            timeout_s = 30.0
            deadline = time.monotonic() + timeout_s
            while not self._async_q.empty() and not self._closed:
                if time.monotonic() > deadline:
                    break
                time.sleep(0.005)
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        if self._closed:
            return
        if self._async_mode:
            self._async_shutdown()
        with self._lock:
            self._flush_locked()
            self.conn.close()
            self._closed = True

    def num_events(self) -> int:
        self.flush()
        return self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def query(self, sql: str, *params):
        self.flush()
        return self.conn.execute(sql, params).fetchall()

    def health(self) -> Dict[str, Any]:
        """Return counters useful for production monitoring.

        Production should poll this every minute and alert when any
        non-zero counter increases — a healthy run keeps all 0."""
        out: Dict[str, Any] = {
            "async_mode": self._async_mode,
            "closed": self._closed,
            "counter": self._counter,
            "buffer_size": len(self._buffer),
        }
        if self._async_mode and self._async_q is not None:
            out.update({
                "queue_size": self._async_q.qsize(),
                "queue_capacity": self._async_q.maxsize,
                "dropped": self._async_dropped,
                "serialize_failures": self._async_serialize_failures,
                "db_failures": self._async_db_failures,
                "thread_alive": self._async_thread.is_alive()
                if self._async_thread else False,
            })
        return out

    # ---- internals ----

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        try:
            self.conn.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
                self._buffer,
            )
        except Exception:
            # In async mode this can race with concurrent close; surface as
            # a counter rather than crash training.
            if self._async_mode:
                self._async_db_failures += 1
            else:
                raise
        self._buffer.clear()

    def _async_drain_loop(self) -> None:
        """Run in the background thread. Pulls pending events off the
        queue, serialises payloads (json.dumps), batches into _buffer,
        flushes to DuckDB. Exits when `_async_done` is set AND queue is
        empty."""
        local_buf: list = []
        while not (self._async_done.is_set() and self._async_q.empty()):
            try:
                pending = self._async_q.get(timeout=self._async_drain_interval)
            except queue.Empty:
                pending = None
            if pending is not None:
                local_buf.append(self._serialise(pending))
                # Drain more without blocking, up to buffer_limit
                while len(local_buf) < self._buffer_limit:
                    try:
                        more = self._async_q.get_nowait()
                    except queue.Empty:
                        break
                    local_buf.append(self._serialise(more))
            if local_buf:
                with self._lock:
                    if not self._closed:
                        try:
                            self.conn.executemany(
                                "INSERT INTO events VALUES "
                                "(?, ?, ?, ?, ?, ?, ?)",
                                local_buf,
                            )
                        except Exception:
                            self._async_db_failures += 1
                local_buf.clear()

    def _serialise(self, pending: _PendingEvent) -> Tuple:
        eid, step, rank, hp, ts, payload = pending
        try:
            pl_str = json.dumps(payload, default=_json_default)
        except Exception as e:  # noqa: BLE001
            self._async_serialize_failures += 1
            pl_str = json.dumps({"_async_serialize_error": str(e)})
        return (eid, step, rank, hp, ts, SCHEMA_VERSION, pl_str)

    def _async_shutdown(self) -> None:
        if not self._async_mode or self._async_done is None:
            return
        if self._async_done.is_set():
            return  # already shutting down
        self._async_done.set()
        if self._async_thread is not None and self._async_thread.is_alive():
            self._async_thread.join(timeout=5.0)


def _json_default(obj):
    """Best-effort JSON encoder for non-trivial types."""
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)
