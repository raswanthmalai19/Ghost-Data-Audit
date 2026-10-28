"""Operational log parsing with trace-level identity propagation.

The observability exports arrive in six line formats (JSON lines, bracketed,
pipe-delimited, RFC5424 syslog, cloud-runtime wrapped messages and logfmt).
Every line is normalised to ``(ts, level, service, trace_id, event, fields)``.

A lookup usually spans several lines that share a ``trace_id``: the *started*
line carries the subject key (email / user id / hash) and the *completed* line
carries the resource id and its state.  Identifiers are therefore propagated
across the whole trace before observations are emitted.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from model import Observation, system_for_resource
from utils import is_null, norm_email, norm_hash, norm_user, parse_ts, tokens_from_value
from extractors.records import _bool, lookup_to_observation

KV_RE = re.compile(r'([A-Za-z_][\w.]*)=("(?:[^"\\]|\\.)*"|\S+)')

SUBJECT_KEYS = (
    "lookup_key", "match_key", "subject_key", "scan_key", "owner_key", "subject_ref",
    "correlation_key", "matched_value", "customer_email", "customer_reference", "requester_email",
    "owner_email", "email", "subject_reference",
)

SERVICE_SYSTEM = {
    "support_replica": "support_platform",
    "profile_service": "account_service",
    "attachment_preview": "support_platform",
}


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1]
    return v.replace('\\"', '"')


def _kv(text: str) -> dict:
    return {k: _unquote(v) for k, v in KV_RE.findall(text)}


def parse_line(line: str) -> Optional[dict]:
    line = line.rstrip("\n")
    m = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(.*)$", line)
    if not m:
        return None
    ts, rest = parse_ts(m.group(1)), m.group(2)

    if rest.startswith("{"):
        try:
            obj = json.loads(rest)
        except json.JSONDecodeError:
            return None
        fields = {k: str(v) for k, v in obj.items()}
        return {
            "ts": ts, "level": fields.pop("severity", ""), "service": fields.get("service") or fields.get("logger", ""),
            "trace": fields.pop("trace_id", ""), "event": fields.pop("event", ""), "fields": fields,
        }

    m2 = re.match(r"^\[(\w+)\]\s+\[([^\]]+)\]\s+\[([^\]]+)\]\s+(\w+):\s*(.*)$", rest)
    if m2:
        return {"ts": ts, "level": m2.group(1), "service": m2.group(2), "trace": m2.group(3),
                "event": m2.group(4), "fields": _kv(m2.group(5))}

    m3 = re.match(r"^\|\s*(\w+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\w+)\s*\|\s*(.*)$", rest)
    if m3:
        return {"ts": ts, "level": m3.group(1), "service": m3.group(2), "trace": m3.group(3),
                "event": m3.group(4), "fields": _kv(m3.group(5))}

    m4 = re.match(r"^<\d+>1\s+(\S+)\s+(\S+)\s+(\S+)\s+-\s+-\s+(.*)$", rest)
    if m4:
        fields = _kv(m4.group(4))
        return {"ts": ts, "level": fields.pop("level", ""), "service": m4.group(2), "trace": m4.group(3),
                "event": fields.pop("event", ""), "fields": fields}

    m5 = re.match(r'^requestId=(\S+)\s+level=(\w+)\s+function=(\S+)\s+message="(.*)"$', rest)
    if m5:
        fields = _kv(m5.group(4).replace('\\"', '"'))
        return {"ts": ts, "level": m5.group(2), "service": m5.group(3), "trace": m5.group(1),
                "event": fields.pop("event", ""), "fields": fields}

    # logfmt, optionally preceded by a bare level word ("INFO service=...")
    m6 = re.match(r"^(DEBUG|INFO|WARN|WARNING|ERROR)\s+(.*)$", rest)
    level = ""
    if m6:
        level, rest = m6.group(1), m6.group(2)
    fields = _kv(rest)
    return {
        "ts": ts, "level": fields.pop("level", level), "service": fields.get("service", ""),
        "trace": fields.pop("trace_id", ""), "event": fields.pop("event", ""), "fields": fields,
    }


def _line_tokens(fields: dict) -> set[str]:
    out: set[str] = set()
    for k in SUBJECT_KEYS:
        v = fields.get(k)
        if v is not None and not is_null(v):
            out.update(tokens_from_value(v))
    return out


def _is_trace(trace: str) -> bool:
    return bool(trace) and (trace.startswith("tr-") or trace.startswith("rb-"))


def _interpret(event: str, f: dict, system: str) -> Optional[dict]:
    """Translate a result-bearing event into observation attributes."""
    def nn(key):
        v = f.get(key)
        return None if v is None or is_null(v) else v

    pii = {}
    for src, dst in (("email", "email"), ("requester_email", "email"), ("customer_email", "email"),
                     ("owner_email", "email"), ("phone", "phone"), ("requester_phone", "phone"),
                     ("requester_name", "name"), ("email_hash", "email_hash")):
        val = nn(src)
        if val:
            pii.setdefault(dst, val)
    ref = norm_user(nn("customer_reference") or nn("subject_reference") or nn("subject_ref") or "")

    if event == "profile_lookup_completed":
        rows, status = f.get("rows"), f.get("http_status")
        present = not (rows == "0" or status == "404")
        return {"category": "USER_PROFILE", "present": present, "pii": pii if present else {},
                "pii_fields_known": "email" in f or not present, "subject_ref": ref}
    if event == "index_query_completed":
        hit = f.get("result", "hit") == "hit" and f.get("total_hits", "1") != "0"
        return {"category": "SEARCH_INDEX_ENTRY", "present": hit, "pii": pii if hit else {},
                "pii_fields_known": "email" in f or not hit, "subject_ref": ref}
    if event in {"message_pii_scan", "attachment_pii_scan"}:
        val = nn("matched_value")
        if val:
            kind = "email" if norm_email(val) else ("email_hash" if norm_hash(val) else "phone")
            pii.setdefault(kind, val)
        cat = "SUPPORT_ATTACHMENT" if event == "attachment_pii_scan" or nn("file") else "SUPPORT_MESSAGE"
        return {"category": cat, "present": True, "pii": pii, "pii_fields_known": True, "subject_ref": ref,
                "extra": {"file_name": nn("file")} if nn("file") else {}}
    if event == "message_read":
        redacted = (f.get("message_text") or "").strip() == "[REDACTED]"
        return {"category": "SUPPORT_MESSAGE", "present": True, "pii": {} if redacted else pii,
                "pii_fields_known": True, "subject_ref": ref, "extra": {"message_redacted": redacted}}
    if event == "support_ticket_read_completed":
        return {"category": "SUPPORT_MESSAGE", "present": True, "pii": pii, "pii_fields_known": True,
                "subject_ref": ref}
    if event == "contact_query_completed":
        present = f.get("rows", "1") != "0"
        return {"category": None, "present": present, "pii": {}, "pii_fields_known": False, "subject_ref": ref}
    if event == "deletion_marker_check":
        found = _bool(f.get("marker_found"))
        return {"category": None, "present": None if found is None else (not found), "pii": {},
                "pii_fields_known": False, "subject_ref": ref}
    if event == "analytics_identity_scan":
        val = nn("matched_value")
        if val:
            kind = "email" if norm_email(val) else ("email_hash" if norm_hash(val) else "phone")
            pii.setdefault(kind, val)
        identifiable = f.get("direct_identifiers", "1") != "0"
        return {"category": "ANALYTICS_PROFILE", "present": True, "pii": pii if identifiable else {},
                "pii_fields_known": True, "subject_ref": ref}
    if event in {"analytics_profile_read", "analytics_profile_read_completed"}:
        return {"category": "ANALYTICS_PROFILE", "present": True, "pii": pii, "pii_fields_known": True,
                "subject_ref": ref}
    if event == "contact_lookup_completed":
        present = f.get("rows", "1") != "0"
        key = nn("correlation_key")
        if key and norm_hash(key):
            pii["email_hash"] = key
        return {"category": None, "present": present, "pii": pii if present else {}, "pii_fields_known": True,
                "subject_ref": ref, "extra": {"email_opt_in": nn("email_opt_in")}}
    if event == "audit_record_read_completed":
        present = f.get("rows", "1") != "0"
        return {"category": "SECURITY_AUDIT_LOG", "present": present, "pii": {}, "pii_fields_known": True,
                "subject_ref": ref, "extra": {"retention_until": nn("retention_until")}}
    if event in {"connector_config_read_completed", "connector_config_loaded", "connector_heartbeat"}:
        enabled = _bool(f.get("enabled"))
        if enabled is None and f.get("scheduler_registered") is not None:
            enabled = _bool(f.get("scheduler_registered"))
        return {"category": "DATA_FLOW", "present": True, "enabled": enabled, "pii": pii,
                "pii_fields_known": "customer_email" in f, "subject_ref": ref}
    if event == "restore_candidate_evaluated":
        eligible = _bool(f.get("eligible"))
        blocked = _bool(f.get("deletion_block"))
        allowed = bool(eligible) and not bool(blocked)
        return {"category": "BACKUP_ARCHIVE", "present": True, "restore_allowed": allowed, "pii": {},
                "pii_fields_known": False, "subject_ref": ref}
    if event == "restore_authorization_completed":
        decision = (f.get("decision") or "").lower()
        return {"category": "BACKUP_ARCHIVE", "present": True,
                "restore_allowed": decision not in {"deny", "denied", "blocked"}, "pii": {},
                "pii_fields_known": False, "subject_ref": ref}
    if event == "restore_policy_read_completed":
        return {"category": "BACKUP_ARCHIVE", "present": _bool(f.get("archive_present")) is not False,
                "restore_allowed": _bool(f.get("restore_allowed")), "pii": pii,
                "pii_fields_known": "customer_email" in f, "subject_ref": ref}
    if event in {"export_download_check_completed", "export_metadata_read_completed"}:
        available = _bool(f.get("available"))
        return {"category": "USER_EXPORT", "present": available is not False, "pii": pii,
                "pii_fields_known": "owner_email" in f, "subject_ref": ref,
                "extra": {"download_available": available, "file_name": nn("file")}}
    if event == "retained_order_loaded":
        return {"category": "LEGAL_HOLD_RECORD", "present": True, "pii": {}, "pii_fields_known": True,
                "subject_ref": ref, "extra": {"retention_reason": nn("reason")}}
    return None


def extract_log(path: Path, evidence_id: str, captured_at: datetime):
    parsed = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        p = parse_line(raw)
        if p:
            parsed.append(p)

    trace_tokens: dict[str, set[str]] = {}
    for p in parsed:
        if _is_trace(p["trace"]):
            trace_tokens.setdefault(p["trace"], set()).update(_line_tokens(p["fields"]))

    observations: list[Observation] = []
    for p in parsed:
        f, event = p["fields"], p["event"]
        if event == "record_lookup" and "resource_id" in f:
            observations.append(
                lookup_to_observation(
                    evidence_id=evidence_id, file_captured_at=captured_at, observed_at=p["ts"],
                    system_id=f.get("system") or system_for_resource(f["resource_id"]) or "",
                    resource_id=f["resource_id"], record_type=f.get("record_type", ""),
                    result=f.get("result", ""), subject=f.get("subject_reference"), source=f"log:{path.name}",
                )
            )
            continue
        rid = f.get("resource_id")
        if not rid or is_null(rid):
            continue
        system = system_for_resource(rid) or SERVICE_SYSTEM.get(p["service"], p["service"])
        info = _interpret(event, f, system)
        if info is None:
            continue
        links = set(_line_tokens(f))
        if _is_trace(p["trace"]):
            links |= trace_tokens.get(p["trace"], set())
        pii = {}
        for k, v in (info.get("pii") or {}).items():
            pii[k] = v
        obs = Observation(
            evidence_id=evidence_id,
            file_captured_at=captured_at,
            observed_at=p["ts"],
            system_id=system,
            resource_id=rid,
            category=info.get("category"),
            present=info.get("present"),
            pii=pii,
            pii_fields_known=bool(info.get("pii_fields_known")),
            subject_ref=info.get("subject_ref"),
            enabled=info.get("enabled"),
            restore_allowed=info.get("restore_allowed"),
            status=event,
            links=links,
            source=f"log:{path.name}:{event}",
            extra={k: v for k, v in (info.get("extra") or {}).items() if v not in (None, "")},
        )
        obs.extra["trace_id"] = p["trace"]
        observations.append(obs)
    return observations, []


__all__ = ["extract_log", "parse_line"]
