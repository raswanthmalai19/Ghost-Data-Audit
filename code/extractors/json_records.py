"""JSON exports and plain-text handover records.

Supported shapes (detected structurally, never by file name):

* snapshot bundles      ``{"snapshots": [{captured_at, data_store, collection, records}]}``
* service exports       ``{"service_exports": {system: {collection: [records with observed_at]}}}``
* lookup ledgers        ``{"records": [{recorded_at, system, resource_id, record_type, result, subject_reference}]}``
* collection captures   ``{"captured_at", "collection", "records": [...]}``
* text handovers        ``- <ts> | <system> | <resource> | <TYPE> | <RESULT> | <subject>``
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from model import COLLECTION_CATEGORY, Observation, SnapshotCoverage
from utils import parse_ts
from extractors.records import lookup_to_observation, record_to_observation

COLLECTION_SYSTEM = {
    "customer_analytics_reports": "analytics_platform",
    "customer_export_jobs": "export_service",
    "customer_profiles": "account_service",
    "customer_search_documents": "search_service",
    "support_conversations": "support_platform",
    "ticket_attachments": "support_platform",
    "connector_configurations": "integration_service",
    "backup_catalog": "backup_service",
    "security_audit_events": "security_service",
    "deleted_customer_references": "identity_service",
    "retained_order_records": "legal_service",
}


def _lookup_rows(records, evidence_id, captured_at, default_time, source):
    out = []
    for r in records:
        if not isinstance(r, dict) or "resource_id" not in r or "system" not in r:
            continue
        observed = parse_ts(r.get("recorded_at")) or default_time
        out.append(
            lookup_to_observation(
                evidence_id=evidence_id, file_captured_at=captured_at, observed_at=observed,
                system_id=r["system"], resource_id=r["resource_id"],
                record_type=r.get("record_type", ""), result=r.get("result", ""),
                subject=r.get("subject_reference") or r.get("lookup_value"), source=source,
            )
        )
    return out


def extract_json(path: Path, evidence_id: str, captured_at: datetime):
    data = json.loads(path.read_text(encoding="utf-8"))
    observations: list[Observation] = []
    coverage: list[SnapshotCoverage] = []

    if isinstance(data.get("snapshots"), list):
        for snap in data["snapshots"]:
            snap_time = parse_ts(snap.get("captured_at")) or captured_at
            system = snap.get("data_store")
            collection = snap.get("collection", "")
            records = snap.get("records") or []
            if collection == "operational_record_history":
                observations.extend(
                    _lookup_rows(records, evidence_id, captured_at, snap_time, f"json:{system}:{collection}")
                )
                continue
            ids = set()
            for rec in records:
                obs = record_to_observation(
                    evidence_id=evidence_id, file_captured_at=captured_at, observed_at=snap_time,
                    system_id=system, collection=collection, record=rec, source=f"json:{collection}",
                )
                if obs:
                    observations.append(obs)
                    ids.add(obs.resource_id)
            coverage.append(
                SnapshotCoverage(evidence_id, captured_at, snap_time, system, ids, f"json:{collection}")
            )
        return observations, coverage

    if isinstance(data.get("service_exports"), dict):
        default_time = parse_ts(data.get("exported_at")) or captured_at
        for system, collections in data["service_exports"].items():
            for collection, records in (collections or {}).items():
                for rec in records or []:
                    observed = parse_ts(rec.get("observed_at")) or default_time
                    obs = record_to_observation(
                        evidence_id=evidence_id, file_captured_at=captured_at, observed_at=observed,
                        system_id=system, collection=collection, record=rec,
                        source=f"json:service_export:{collection}",
                    )
                    if obs:
                        observations.append(obs)
        return observations, coverage

    records = data.get("records")
    if isinstance(records, list) and records and isinstance(records[0], dict) and "record_type" in records[0]:
        default_time = parse_ts(data.get("exported_at")) or captured_at
        observations.extend(_lookup_rows(records, evidence_id, captured_at, default_time, "json:lookup_ledger"))
        return observations, coverage

    if isinstance(records, list) and data.get("collection"):
        collection = data["collection"]
        system = COLLECTION_SYSTEM.get(collection)
        snap_time = parse_ts(data.get("captured_at")) or captured_at
        if system:
            ids = set()
            for rec in records:
                obs = record_to_observation(
                    evidence_id=evidence_id, file_captured_at=captured_at, observed_at=snap_time,
                    system_id=system, collection=collection, record=rec, source=f"json:{collection}",
                )
                if obs:
                    observations.append(obs)
                    ids.add(obs.resource_id)
            coverage.append(SnapshotCoverage(evidence_id, captured_at, snap_time, system, ids, f"json:{collection}"))
    return observations, coverage


HANDOVER_LINE = re.compile(
    r"^-\s*(?P<ts>\S+)\s*\|\s*(?P<system>[a-z_]+)\s*\|\s*(?P<rid>[a-z_]+_\d+)\s*\|\s*(?P<type>[A-Z_]+)"
    r"\s*\|\s*(?P<result>[A-Z_]+)\s*\|\s*(?P<subject>.*)$"
)


def extract_text(path: Path, evidence_id: str, captured_at: datetime):
    observations: list[Observation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = HANDOVER_LINE.match(line.strip())
        if not m:
            continue
        observations.append(
            lookup_to_observation(
                evidence_id=evidence_id, file_captured_at=captured_at,
                observed_at=parse_ts(m.group("ts")) or captured_at,
                system_id=m.group("system"), resource_id=m.group("rid"),
                record_type=m.group("type"), result=m.group("result"),
                subject=m.group("subject").strip(), source="text:handover",
            )
        )
    return observations, []


__all__ = ["extract_json", "extract_text", "COLLECTION_CATEGORY"]
