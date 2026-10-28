"""Shared helpers: paths, timestamp parsing and identifier normalisation.

Everything in the audit pipeline compares identifiers through the normalisers
defined here so that ``LDAHL9787@FASTMAIL.COM`` and ``ldahl9787@fastmail.com``
resolve to the same subject while near-miss decoys such as
``smartinez6461.jr@gmail.com`` stay distinct.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = CODE_DIR / "cache"

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
USER_RE = re.compile(r"\buser_\d+\b", re.IGNORECASE)
HASH_RE = re.compile(r"\b[0-9a-f]{64}\b", re.IGNORECASE)
RESOURCE_RE = re.compile(
    r"\b(?:account|search|support|export|analytics|backup|data_flow|marketing|tombstone|"
    r"legal_hold|object|audit_record|cache|email_delivery|telemetry)_\d+\b"
)


def parse_ts(value: Optional[str]) -> Optional[datetime]:
    """Parse the ISO-8601 UTC timestamps used throughout the dataset."""
    if not value:
        return None
    value = value.strip()
    for fmt in (ISO_FMT, "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    return None


def fmt_ts(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime(ISO_FMT) if dt else None


def norm_email(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip().strip('"').strip("'").lower()
    return value if EMAIL_RE.fullmatch(value) else None


def norm_user(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    m = USER_RE.fullmatch(value.strip())
    return m.group(0).lower() if m else None


def norm_phone(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) >= 7 else None


def norm_hash(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def norm_name(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value or None


def email_hash(email: str) -> str:
    """The dataset stores ``email_hash`` as SHA-256 of the lower-case email."""
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()


def token(kind: str, value: Optional[str]) -> Optional[str]:
    """Build a typed identifier token (``email:``, ``user:``, ``hash:``, ``phone:``)."""
    if value is None:
        return None
    normaliser = {
        "email": norm_email,
        "user": norm_user,
        "hash": norm_hash,
        "phone": norm_phone,
        "name": norm_name,
    }[kind]
    normalised = normaliser(value)
    return f"{kind}:{normalised}" if normalised else None


def tokens_from_value(value) -> set[str]:
    """Classify a free-form identifier value (lookup keys, selectors, subjects)."""
    out: set[str] = set()
    if value is None or not isinstance(value, str):
        return out
    raw = value.strip().strip('"')
    if not raw or raw.lower() in {"null", "none", "-", "—"}:
        return out
    for kind in ("email", "user", "hash"):
        tok = token(kind, raw)
        if tok:
            out.add(tok)
            return out
    if re.fullmatch(r"\+?[\d\s().\-]{7,}", raw):
        tok = token("phone", raw)
        if tok:
            out.add(tok)
    return out


def tokens_in_text(text: str) -> set[str]:
    """Extract every email / user id / hash token mentioned inside free text."""
    out: set[str] = set()
    if not text:
        return out
    for m in EMAIL_RE.finditer(text):
        tok = token("email", m.group(0))
        if tok:
            out.add(tok)
    for m in USER_RE.finditer(text):
        tok = token("user", m.group(0))
        if tok:
            out.add(tok)
    for m in HASH_RE.finditer(text):
        tok = token("hash", m.group(0))
        if tok:
            out.add(tok)
    return out


def is_null(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none", "-", "—"}:
        return True
    return False


def first_non_null(values: Iterable):
    for v in values:
        if not is_null(v):
            return v
    return None
