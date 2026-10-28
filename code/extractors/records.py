"""Normalise structured records (SQL rows, JSON objects, lookup rows) into observations.

The same customer record shows up with different field names depending on the
service that exported it (``email`` / ``owner_email`` / ``requester_email`` /
``customer_email`` / ``subject_selector`` ...).  This module maps all of them to
one schema so the audit engine can reason about presence, exposed identifiers,
stable references, data-flow state and restore state uniformly.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from model import COLLECTION_CATEGORY, RECORD_TYPE_CATEGORY, SYSTEM_DEFAULT_CATEGORY, Observation
from utils import EMAIL_RE, is_null, norm_email, norm_hash, norm_user, token, tokens_from_value

PHONE_IN_TEXT = re.compile(r"\+\d[\d\s().\-]{6,}\d")

EMAIL_FIELDS = ("email", "owner_email", "requester_email", "customer_email", "uploaded_by")
NAME_FIELDS = ("full_name", "name", "display_name", "requester_name")
PHONE_FIELDS = ("phone", "owner_phone", "requester_phone")
REF_FIELDS = ("customer_reference", "subject_reference", "user_id")
TEXT_FIELDS = ("message_text", "message_excerpt")


def _bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().strip('"').lower()
    if s in {"1", "true", "yes", "enabled", "allowed"}:
        return True
    if s in {"0", "false", "no", "disabled", "denied", "deny"}:
        return False
    return None


def flatten_record(record: dict) -> dict:
    """Lift nested ``document`` / ``profile`` payloads to the top level."""
    flat = dict(record)
    for nested in ("document", "profile"):
        inner = record.get(nested)
        if isinstance(inner, dict):
            for k, v in inner.items():
                flat.setdefault(k, v)
                if k in flat and is_null(flat[k]) and not is_null(v):
                    flat[k] = v
    return flat


def record_to_observation(
    *,
    evidence_id: str,
    file_captured_at: datetime,
    observed_at: datetime,
    system_id: str,
    collection: str,
    record: dict,
    source: str,
    replica: Optional[str] = None,
) -> Optional[Observation]:
    rec = flatten_record(record)
    rid = rec.get("resource_id")
    if is_null(rid):
        return None
    rid = str(rid).strip()
    category = COLLECTION_CATEGORY.get(collection) or SYSTEM_DEFAULT_CATEGORY.get(system_id)

    pii: dict = {}
    links: set[str] = set()
    names = [rec.get(f) for f in NAME_FIELDS if f in rec and not is_null(rec.get(f))]
    if names:
        pii["name"] = str(names[0])
    for f in EMAIL_FIELDS:
        if f in rec and not is_null(rec.get(f)):
            e = norm_email(str(rec[f]))
            if e:
                pii.setdefault("email", e)
                links.add(f"email:{e}")
    for f in PHONE_FIELDS:
        if f in rec and not is_null(rec.get(f)):
            pii.setdefault("phone", str(rec[f]))
            tok = token("phone", str(rec[f]))
            if tok:
                links.add(tok)
    if "email_hash" in rec and not is_null(rec.get("email_hash")):
        h = norm_hash(str(rec["email_hash"]))
        if h:
            pii["email_hash"] = h
            links.add(f"hash:{h}")

    subject_ref = None
    for f in REF_FIELDS:
        if f in rec and not is_null(rec.get(f)):
            u = norm_user(str(rec[f]))
            if u:
                subject_ref = u
                links.add(f"user:{u}")
                break
            # some exports put an email in subject_reference
            links.update(tokens_from_value(str(rec[f])))
    selector = rec.get("subject_selector")
    if not is_null(selector):
        sel_tokens = tokens_from_value(str(selector))
        links.update(sel_tokens)
        for t in sel_tokens:
            if t.startswith("user:") and subject_ref is None:
                subject_ref = t.split(":", 1)[1]
            if t.startswith("email:"):
                pii.setdefault("email", t.split(":", 1)[1])
    for f in ("lookup_value", "query"):
        if f in rec and not is_null(rec.get(f)):
            links.update(tokens_from_value(str(rec[f])))

    text_has_pii = False
    for f in TEXT_FIELDS:
        text = rec.get(f)
        if isinstance(text, str) and text and "[REDACTED]" not in text:
            for m in EMAIL_RE.finditer(text):
                e = norm_email(m.group(0))
                if e:
                    pii.setdefault("email", e)
                    links.add(f"email:{e}")
                    text_has_pii = True
            for m in PHONE_IN_TEXT.finditer(text):
                pii.setdefault("phone", m.group(0))
                tok = token("phone", m.group(0))
                if tok:
                    links.add(tok)
                text_has_pii = True

    hits = rec.get("hits")
    present: Optional[bool] = True
    if isinstance(hits, dict):
        docs = hits.get("documents") or []
        total = hits.get("total")
        present = bool(docs) or bool(total)
        for d in docs:
            if isinstance(d, dict) and str(d.get("resource_id", rid)) == rid:
                sub = record_to_observation(
                    evidence_id=evidence_id, file_captured_at=file_captured_at, observed_at=observed_at,
                    system_id=system_id, collection=collection, record=d, source=source, replica=replica,
                )
                if sub:
                    pii.update(sub.pii)
                    links.update(sub.links)
                    subject_ref = subject_ref or sub.subject_ref
    status_code = rec.get("status_code")
    if status_code is not None:
        try:
            present = int(status_code) < 400
        except (TypeError, ValueError):
            pass

    status = rec.get("account_status")
    enabled = _bool(rec.get("enabled")) if "enabled" in rec else None
    restore_allowed = _bool(rec.get("restore_allowed")) if "restore_allowed" in rec else None
    if "download_available" in rec:
        extra_available = _bool(rec.get("download_available"))
    elif "available" in rec:
        extra_available = _bool(rec.get("available"))
    else:
        extra_available = None

    if not present:
        pii = {}

    return Observation(
        evidence_id=evidence_id,
        file_captured_at=file_captured_at,
        observed_at=observed_at,
        system_id=system_id,
        resource_id=rid,
        category=category,
        present=present,
        pii=pii,
        pii_fields_known=True,
        subject_ref=subject_ref,
        enabled=enabled,
        restore_allowed=restore_allowed,
        status=str(status) if status else None,
        links=links,
        source=source,
        replica=replica,
        extra={
            k: v
            for k, v in {
                "download_available": extra_available,
                "retention_until": rec.get("retention_until") or rec.get("retained_until"),
                "message_redacted": any(
                    isinstance(rec.get(f), str) and "[REDACTED]" in rec.get(f) for f in TEXT_FIELDS
                ),
                "text_has_pii": text_has_pii,
                "connector_name": rec.get("connector_name"),
                "destination": rec.get("destination"),
                "file_name": rec.get("file_name"),
                "source_resource_id": rec.get("source_resource_id"),
            }.items()
            if v not in (None, False, "")
        },
    )


# ---------------------------------------------------------------------------
# Lookup-style rows: (system, resource_id, record_type, result, subject)
# ---------------------------------------------------------------------------

def lookup_to_observation(
    *,
    evidence_id: str,
    file_captured_at: datetime,
    observed_at: datetime,
    system_id: str,
    resource_id: str,
    record_type: str,
    result: str,
    subject: Optional[str],
    source: str,
) -> Observation:
    result = (result or "").strip().upper()
    record_type = (record_type or "").strip().upper()
    category = RECORD_TYPE_CATEGORY.get(record_type) or SYSTEM_DEFAULT_CATEGORY.get(system_id)
    links = tokens_from_value(subject) if subject else set()
    subject_ref = None
    pii: dict = {}
    for t in links:
        kind, _, value = t.partition(":")
        if kind == "user":
            subject_ref = value
        elif kind == "email":
            pii["email"] = value
        elif kind == "hash":
            pii["email_hash"] = value
        elif kind == "phone":
            pii["phone"] = value

    obs = Observation(
        evidence_id=evidence_id,
        file_captured_at=file_captured_at,
        observed_at=observed_at,
        system_id=system_id,
        resource_id=resource_id,
        category=category,
        status=result,
        links=links,
        subject_ref=subject_ref,
        source=source,
    )
    if result in {"NOT_FOUND", "ABSENT", "DELETED", "MISSING"}:
        obs.present = False
        obs.pii_fields_known = True
    elif result in {"FOUND", "PRESENT", "IDENTIFIABLE", "EXPOSED"}:
        obs.present = True
        obs.pii = pii
        # FOUND on a record keyed by an email means the copy still carries it.
        obs.pii_fields_known = bool(pii) or result == "IDENTIFIABLE"
    elif result in {"ANONYMIZED", "REDACTED"}:
        obs.present = True
        obs.pii = {}
        obs.pii_fields_known = True
    elif result in {"ACTIVE", "ENABLED"}:
        obs.present = True
        if category == "DATA_FLOW":
            obs.enabled = True
        elif category == "BACKUP_ARCHIVE":
            obs.restore_allowed = True
    elif result in {"DISABLED", "STOPPED"}:
        obs.present = True
        obs.enabled = False
    elif result in {"BLOCKED"}:
        obs.present = True
        obs.restore_allowed = False
    return obs
