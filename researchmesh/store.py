import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from .contracts import canonical, decode, digest, fail, integer

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY,priority INTEGER NOT NULL,generation INTEGER NOT NULL,
 revision INTEGER NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dictionary(kind TEXT NOT NULL,alias TEXT NOT NULL,target TEXT NOT NULL,
 PRIMARY KEY(kind,alias));
CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,source TEXT NOT NULL,local_id TEXT NOT NULL,
 generation INTEGER NOT NULL,revision INTEGER NOT NULL,active INTEGER NOT NULL,
 raw TEXT NOT NULL,raw_digest TEXT NOT NULL,normalized TEXT NOT NULL,norm_digest TEXT NOT NULL,dict_revision INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS record_source ON records(source,active,key);
CREATE TABLE IF NOT EXISTS history(key TEXT NOT NULL,revision INTEGER NOT NULL,raw TEXT NOT NULL,
 raw_digest TEXT NOT NULL,generation INTEGER NOT NULL,active INTEGER NOT NULL,PRIMARY KEY(key,revision));
CREATE TABLE IF NOT EXISTS batches(id TEXT PRIMARY KEY,source TEXT NOT NULL,generation INTEGER NOT NULL,
 digest TEXT NOT NULL,result TEXT NOT NULL,UNIQUE(source,generation));
CREATE TABLE IF NOT EXISTS blocks(block TEXT NOT NULL,record TEXT NOT NULL,PRIMARY KEY(block,record));
CREATE INDEX IF NOT EXISTS block_record ON blocks(record);
CREATE TABLE IF NOT EXISTS decisions(id TEXT PRIMARY KEY,left_key TEXT NOT NULL,right_key TEXT NOT NULL,
 kind TEXT NOT NULL,state TEXT NOT NULL,evidence TEXT NOT NULL,digest TEXT NOT NULL,reason TEXT NOT NULL,operator TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS components(record TEXT PRIMARY KEY,cluster TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS component_cluster ON components(cluster,record);
CREATE TABLE IF NOT EXISTS pins(cluster TEXT NOT NULL,field TEXT NOT NULL,record TEXT NOT NULL,
 value_digest TEXT NOT NULL,reason TEXT NOT NULL,revision INTEGER NOT NULL,PRIMARY KEY(cluster,field));
CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_tasks(id TEXT PRIMARY KEY,left_key TEXT NOT NULL,right_key TEXT NOT NULL,
 evidence TEXT NOT NULL,digest TEXT NOT NULL,state TEXT NOT NULL,revision INTEGER NOT NULL,
 owner TEXT,lease_until INTEGER NOT NULL,result TEXT);
CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,entity TEXT NOT NULL,
 payload TEXT NOT NULL,previous TEXT NOT NULL,digest TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS calibration_labels(id TEXT PRIMARY KEY,pair_id TEXT NOT NULL,left_key TEXT NOT NULL,
 right_key TEXT NOT NULL,label TEXT NOT NULL,source TEXT NOT NULL,note TEXT NOT NULL,
 state TEXT NOT NULL,recorded_at INTEGER NOT NULL,withdrawn_at INTEGER,withdraw_reason TEXT,
 supersedes TEXT,payload TEXT NOT NULL,digest TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS label_pair_state ON calibration_labels(pair_id,state);
CREATE TABLE IF NOT EXISTS calibration_freeze(id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL,
 created_at INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS calibration_experiments(id TEXT PRIMARY KEY,freeze_id TEXT NOT NULL,
 role TEXT NOT NULL,method TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,created_at INTEGER NOT NULL,
 UNIQUE(freeze_id,role));
CREATE TABLE IF NOT EXISTS calibration_suggestions(id TEXT PRIMARY KEY,batch_digest TEXT NOT NULL,
 experiment_id TEXT NOT NULL,payload TEXT NOT NULL,digest TEXT NOT NULL,created_at INTEGER NOT NULL,
 UNIQUE(batch_digest));
"""


class Store:
    def __init__(self, path=":memory:"):
        if str(path) != ":memory:" and Path(path).is_symlink():
            fail("path", "database may not be a symlink")
        self.lock = threading.RLock()
        self.depth = 0
        self.connection = sqlite3.connect(
            path, isolation_level=None, timeout=10, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self.connection.executescript(SCHEMA)
        version = self.one("SELECT value FROM meta WHERE key='schema'")
        if version and version["value"] != "1":
            self.connection.close()
            fail("schema", "unsupported master-data schema")
        for name, value in [
            ("schema", "1"),
            ("dictionary_revision", "0"),
            ("graph_revision", "0"),
            ("data_revision", "0"),
        ]:
            self.connection.execute(
                "INSERT OR IGNORE INTO meta VALUES(?,?)", (name, value)
            )

    @contextmanager
    def atomic(self):
        with self.lock:
            level = self.depth
            self.connection.execute(
                "BEGIN IMMEDIATE" if level == 0 else f"SAVEPOINT mesh{level}"
            )
            self.depth += 1
            try:
                yield self.connection
            except BaseException:
                if level == 0:
                    self.connection.execute("ROLLBACK")
                else:
                    self.connection.execute(f"ROLLBACK TO mesh{level}")
                    self.connection.execute(f"RELEASE mesh{level}")
                raise
            else:
                self.connection.execute(
                    "COMMIT" if level == 0 else f"RELEASE mesh{level}"
                )
            finally:
                self.depth -= 1

    def rows(self, sql, args=()):
        with self.lock:
            return [dict(row) for row in self.connection.execute(sql, args)]

    def one(self, sql, args=()):
        rows = self.rows(sql, args)
        return rows[0] if rows else None

    def revision(self, name):
        if name not in {"dictionary", "graph", "data"}:
            fail("input", "unknown revision counter")
        return int(
            self.one("SELECT value FROM meta WHERE key=?", (name + "_revision",))[
                "value"
            ]
        )

    def bump(self, name):
        if self.depth == 0:
            fail("state", "revision increment requires transaction")
        value = self.revision(name) + 1
        self.connection.execute(
            "UPDATE meta SET value=? WHERE key=?", (str(value), name + "_revision")
        )
        return value

    def audit(self, kind, entity, payload):
        if self.depth == 0:
            fail("state", "audit requires transaction")
        last = self.one("SELECT seq,digest FROM audit ORDER BY seq DESC LIMIT 1")
        seq = last["seq"] + 1 if last else 1
        previous = last["digest"] if last else "0" * 64
        checksum = digest(
            {
                "seq": seq,
                "kind": kind,
                "entity": entity,
                "payload": payload,
                "previous": previous,
            }
        )
        self.connection.execute(
            "INSERT INTO audit VALUES(?,?,?,?,?,?)",
            (seq, kind, entity, canonical(payload), previous, checksum),
        )
        return seq

    def verify_audit(self):
        previous = "0" * 64
        rows = self.rows("SELECT * FROM audit ORDER BY seq")
        for seq, row in enumerate(rows, 1):
            payload = decode(row["payload"])
            checksum = digest(
                {
                    "seq": seq,
                    "kind": row["kind"],
                    "entity": row["entity"],
                    "payload": payload,
                    "previous": previous,
                }
            )
            if (
                row["seq"] != seq
                or row["previous"] != previous
                or row["digest"] != checksum
            ):
                fail("integrity", "audit sequence changed", sequence=seq)
            previous = checksum
        return {"verified": True, "count": len(rows), "head": previous}

    def audit_page(self, after=0, limit=100):
        integer(after)
        integer(limit, 1, 1000)
        rows = self.rows(
            "SELECT * FROM audit WHERE seq>? ORDER BY seq LIMIT ?", (after, limit + 1)
        )
        page = rows[:limit]
        for row in page:
            row["payload"] = decode(row["payload"])
        return {"rows": page, "next": page[-1]["seq"] if len(rows) > limit else None}

    def backup(self, path):
        path = Path(path)
        if path.exists() or path.is_symlink():
            fail("path", "backup destination must be new")
        with self.lock:
            if self.depth:
                fail("state", "backup requires no active transaction")
            target = sqlite3.connect(path)
            try:
                self.connection.backup(target)
            finally:
                target.close()
        return {"path": str(path), "bytes": path.stat().st_size}

    def close(self):
        if self.depth:
            fail("state", "active transaction cannot close")
        self.connection.close()
