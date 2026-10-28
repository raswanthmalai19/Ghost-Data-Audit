"""SQLite-compatible snapshot exports (primary, read / regional / reporting / failover replicas, inventories).

Each file is executed in an in-memory SQLite database so the evidence is read
exactly as a DBA would.  Every row in every table becomes an observation tagged
with the table (replica) it came from, which is how replication drift becomes
visible: a record deleted on the primary but still present on a replica is a
remaining copy.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from model import COLLECTION_CATEGORY, Observation, SnapshotCoverage
from utils import parse_ts
from extractors.records import record_to_observation

INVENTORY_TABLE_SYSTEM = {
    "customer_profiles": "account_service",
    "customer_search_documents": "search_service",
    "customer_analytics_profiles": "analytics_platform",
    "support_records": "support_platform",
    "customer_exports": "export_service",
    "connector_subscriptions": "integration_service",
    "backup_restore_entries": "backup_service",
    "security_audit_records": "security_service",
}


CUSTOMER_DATA_SYSTEMS = (
    "account_service", "search_service", "support_platform", "export_service", "analytics_platform",
    "backup_service", "integration_service", "security_service", "identity_service", "legal_service",
)


def _replica_name(table: str) -> str:
    for prefix in ("customer_data_", "customer_profiles_", "search_documents_"):
        if table.startswith(prefix):
            return table[len(prefix):]
    return table


def extract_sql(path: Path, evidence_id: str, captured_at: datetime):
    text = path.read_text(encoding="utf-8")
    con = sqlite3.connect(":memory:")
    con.executescript(text)
    m = re.search(r"Captured at:\s*(\S+)", text)
    header_time = parse_ts(m.group(1)) if m else None
    snap_time = header_time or captured_at

    observations: list[Observation] = []
    coverage: list[SnapshotCoverage] = []
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]

    scope: dict[str, datetime] = {}
    if "snapshot_scope" in tables:
        for system, ts in con.execute("select origin_system, captured_at from snapshot_scope"):
            scope[system] = parse_ts(ts) or snap_time

    covered: dict[str, set] = {}
    for table in tables:
        if table == "snapshot_scope":
            continue
        cols = [r[1] for r in con.execute(f'pragma table_info("{table}")')]
        rows = [dict(zip(cols, r)) for r in con.execute(f'select * from "{table}"')]
        replica = _replica_name(table)

        if "record_json" in cols:
            for row in rows:
                system = row["origin_system"]
                record = json.loads(row["record_json"])
                obs = record_to_observation(
                    evidence_id=evidence_id, file_captured_at=captured_at, observed_at=snap_time,
                    system_id=system, collection=row["collection_name"], record=record,
                    source=f"sql:{table}", replica=replica,
                )
                if obs:
                    obs.extra["row_updated_at"] = row.get("updated_at")
                    observations.append(obs)
                    covered.setdefault(system, set()).add(obs.resource_id)
            continue

        if table.startswith("customer_profiles_"):
            system, collection = "account_service", "customer_profiles"
        elif table.startswith("search_documents_"):
            system, collection = "search_service", "customer_search_documents"
        elif table in INVENTORY_TABLE_SYSTEM:
            system, collection = INVENTORY_TABLE_SYSTEM[table], table
        else:
            continue
        observed = scope.get(system, snap_time)
        covered.setdefault(system, set())
        for row in rows:
            record = dict(row)
            if "user_id" in record and "customer_reference" not in record:
                record["customer_reference"] = record.pop("user_id")
            obs = record_to_observation(
                evidence_id=evidence_id, file_captured_at=captured_at, observed_at=observed,
                system_id=system, collection=collection, record=record,
                source=f"sql:{table}", replica=replica,
            )
            if obs:
                obs.extra["row_updated_at"] = row.get("updated_at") or row.get("indexed_at")
                if collection == "customer_profiles" and obs.status == "DELETION_PENDING":
                    obs.extra["deletion_pending"] = True
                observations.append(obs)
                covered[system].add(obs.resource_id)

    # Coverage: inventories declare their scope explicitly; a customer-data dump
    # (record_json tables) is a point-in-time copy of every customer-data
    # collection; dedicated table dumps cover the system their tables belong to.
    if scope:
        systems = set(scope)
    elif any("record_json" in [r[1] for r in con.execute(f'pragma table_info("{t}")')] for t in tables):
        systems = set(CUSTOMER_DATA_SYSTEMS) | set(covered)
    else:
        systems = set(covered)
    for system in systems:
        coverage.append(
            SnapshotCoverage(
                evidence_id=evidence_id,
                file_captured_at=captured_at,
                captured_at=scope.get(system, snap_time),
                system_id=system,
                resource_ids=set(covered.get(system, set())),
                source=f"sql:{path.name}",
            )
        )
    con.close()
    return observations, coverage


__all__ = ["extract_sql", "COLLECTION_CATEGORY"]
