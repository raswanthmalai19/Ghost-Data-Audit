"""PDF documents and application screenshots.

Text is pulled with ``pdfplumber`` (PDF) or Tesseract OCR (PNG/JPG, 2x upscale).
OCR output is noisy (``data_flow1121``, ``areedi433@gmail.com``,
``outiook.com``) so every identifier is *grounded*: snapped to the closest
identifier that already exists in participant data (resource ids from the
activity stream and structured evidence, emails from the identity tables) and
accepted only when the match is unambiguous.  The extractor then emits one
observation per grounded resource with the identifiers that appear on the same
text line and the state implied by the screen (``Continuous``/``Healthy``
connectors are active, ``Retained reference`` registries keep no contact data,
and so on).

An optional VLM pass (``llm.vision_extract``) can replace OCR when an API key is
configured; its JSON output goes through the same grounding step.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from model import Observation, SYSTEM_DEFAULT_CATEGORY, system_for_resource
from utils import CACHE_DIR, norm_email

RESOURCE_PREFIXES = (
    "audit_record", "data_flow", "legal_hold", "email_delivery", "account", "search", "support", "export",
    "analytics", "backup", "marketing", "tombstone", "object", "cache", "telemetry",
)
OCR_RESOURCE_RE = re.compile(
    r"(?P<prefix>audit[\s_.]?record|data[\s_.]?flow|legal[\s_.]?hold|account|search|support|export|analytics|"
    r"backup|marketing|tombstone|object|cache)(?P<sep>[\s_.\-]{0,2})(?P<num>[0-9OoIlS]{1,6})\b",
    re.IGNORECASE,
)
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"})
OCR_EMAIL_RE = re.compile(r"[A-Za-z0-9._,%+\-]+@[A-Za-z0-9\-]+(?:[.,][A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
OCR_USER_RE = re.compile(r"\buser[\s_]{1,2}(\d{1,4})\b", re.IGNORECASE)


def _levenshtein(a: str, b: str, limit: int = 3) -> int:
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            best = min(best, cur[-1])
        if best > limit:
            return limit + 1
        prev = cur
    return prev[-1]


@dataclass
class Grounder:
    """Snaps OCR'd identifiers onto identifiers known from participant data."""

    resource_ids: set
    emails: set
    users: set
    _by_prefix: dict = field(default_factory=dict)

    def __post_init__(self):
        for rid in self.resource_ids:
            prefix = rid.rsplit("_", 1)[0]
            self._by_prefix.setdefault(prefix, set()).add(rid)

    def resource(self, prefix_raw: str, num_raw: str, exact_text: bool = False, sep: str = "_") -> Optional[str]:
        """Canonicalise a resource id.  OCR'd digits are never fuzzily re-mapped onto a
        *different* known id (that would invent evidence); only separator noise and
        letter/digit confusions (O->0, l->1) are repaired."""
        prefix = re.sub(r"[\s_.]+", "_", prefix_raw.lower())
        num = num_raw.translate(DIGIT_FIX)
        if not num.isdigit():
            return None
        candidate = f"{prefix}_{num}"
        if candidate in self.resource_ids or exact_text:
            return candidate
        if num_raw.isdigit() and len(num) >= 3 and sep == "_":
            return candidate  # cleanly read id (prefix_123) not referenced elsewhere
        return None

    def email(self, raw: str) -> Optional[str]:
        e = raw.strip().strip(".,;:").replace(",", ".").lower()
        e = re.sub(r"\.{2,}", ".", e)
        if e in self.emails:
            return e
        close = [k for k in self.emails if _levenshtein(k, e, 2) <= 2]
        if len(close) == 1:
            return close[0]
        return norm_email(e) if not close else None


def _pdf_text(path: Path) -> str:
    import pdfplumber  # local import keeps startup light

    with pdfplumber.open(str(path)) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def _ocr_text(path: Path) -> str:
    """Tesseract OCR with a 2x upscale; dark themes are inverted first."""
    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageStat
    except ImportError as exc:  # keep the pipeline runnable; screenshots then contribute nothing
        print(f"[ocr] {path.name}: OCR unavailable ({exc}); install Pillow + pytesseract + tesseract")
        return ""

    img = Image.open(str(path)).convert("L")
    if ImageStat.Stat(img).mean[0] < 110:
        img = ImageOps.invert(img)
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    try:
        return pytesseract.image_to_string(img, config="--psm 3")
    except pytesseract.TesseractNotFoundError:
        print(f"[ocr] {path.name}: tesseract binary not found; install tesseract-ocr")
        return ""


def document_text(path: Path) -> tuple[str, str]:
    """Return (text, method), caching OCR/PDF output by content hash."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    cache = CACHE_DIR / "documents" / f"{digest}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data["text"], data["method"]
    if path.suffix.lower() == ".pdf":
        text, method = _pdf_text(path), "pdfplumber"
    else:
        text, method = _ocr_text(path), "tesseract"
    if not text.strip():
        return text, method  # never cache a failed extraction
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"file": path.name, "method": method, "text": text}, indent=1), encoding="utf-8")
    return text, method


def _grounded_ids(text: str, grounder: Grounder, exact_text: bool = False) -> list[str]:
    out = []
    for m in OCR_RESOURCE_RE.finditer(text):
        rid = grounder.resource(m.group("prefix"), m.group("num"), exact_text, m.group("sep") or "")
        if rid and rid not in out:
            out.append(rid)
    return out


def _grounded_emails(text: str, grounder: Grounder) -> list[str]:
    out = []
    for m in OCR_EMAIL_RE.finditer(text):
        e = grounder.email(m.group(0))
        if e and e not in out:
            out.append(e)
    return out


def _grounded_users(text: str) -> list[str]:
    return sorted({f"user_{int(m.group(1)):02d}" if len(m.group(1)) < 2 else f"user_{m.group(1)}"
                   for m in OCR_USER_RE.finditer(text)})


STAFF_DOMAINS = ("northstar.shop", "company.com")

STATE_HINTS = {
    "DATA_FLOW": {
        "enabled": ("continuous", "healthy", "active", "sync mode", "events delivered"),
        "disabled": ("disabled", "paused", "stopped", "disconnected"),
    },
    "BACKUP_ARCHIVE": {
        "allowed": ("archive", "restored", "checksum verified", "restore success"),
        "blocked": ("restore blocked", "deletion block", "legal hold", "restore disabled", "deny"),
    },
}


def _vlm_lines(vlm, path: Path, ocr_text: str, grounder: Grounder) -> list[str]:
    """Optional VLM transcription, cross-validated: an identifier is kept only when it is
    already known from participant data or also appears in the OCR text."""
    if vlm is None or path.suffix.lower() == ".pdf":
        return []
    result = vlm.vision_extract(path)
    if not result:
        return []
    squashed = re.sub(r"[^a-z0-9]", "", ocr_text.lower())
    lines = []
    for rec in result.get("records", []):
        rid = str(rec.get("resource_id", "")).strip().lower()
        if not re.fullmatch(r"[a-z_]+_\d+", rid):
            continue
        digits = rid.rsplit("_", 1)[-1]
        on_screen = re.sub(r"[^a-z0-9]", "", rid) in squashed
        if not on_screen and not (rid in grounder.resource_ids and digits in ocr_text):
            continue  # never let the model introduce an identifier the OCR cannot see
        parts = [rid] + [str(e) for e in rec.get("emails", [])] + [str(u) for u in rec.get("user_ids", [])]
        parts.append(str(rec.get("status_text", "")))
        lines.append(" ".join(parts))
    return lines


def extract_document(path: Path, evidence_id: str, captured_at: datetime, grounder: Grounder, vlm=None):
    text, method = document_text(path)
    extra_lines = _vlm_lines(vlm, path, text, grounder)
    if extra_lines:
        text = text + "\n" + "\n".join(extra_lines)
        method = method + "+vlm"
    lower = text.lower()
    observations: list[Observation] = []
    doc_emails = [e for e in _grounded_emails(text, grounder) if not e.endswith(STAFF_DOMAINS)]
    doc_users = _grounded_users(text)
    resources = _grounded_ids(text, grounder, exact_text=(method == "pdfplumber"))
    single_subject = len(set(doc_emails)) <= 1 and len(doc_users) <= 1

    for rid in resources:
        system = system_for_resource(rid)
        if not system:
            continue
        category = SYSTEM_DEFAULT_CATEGORY.get(system)
        if system == "support_platform" and ("attachment" in lower or ".pdf" in lower and "uploader" in lower):
            category = "SUPPORT_ATTACHMENT"
        # identifiers on the same text line as the resource id (a table row); a row that
        # names someone else (including staff accounts) never inherits the page subject
        line_links: set[str] = set()
        line_emails: list[str] = []
        row_names_someone = False
        for line in text.splitlines():
            if rid in _grounded_ids(line, grounder, exact_text=(method == "pdfplumber")):
                found = _grounded_emails(line, grounder)
                row_names_someone |= bool(found)
                line_emails += [e for e in found if not e.endswith(STAFF_DOMAINS)]
                line_links.update(f"user:{u}" for u in _grounded_users(line))
        fallback = single_subject and not row_names_someone
        emails = line_emails or (doc_emails if fallback else [])
        users = [t.split(":", 1)[1] for t in line_links] or (doc_users if fallback else [])
        links = {f"email:{e}" for e in emails} | {f"user:{u}" for u in users}
        pii = {}
        if emails and category not in {"SECURITY_AUDIT_LOG", "ACCOUNT_TOMBSTONE"}:
            pii["email"] = emails[0]
        obs = Observation(
            evidence_id=evidence_id,
            file_captured_at=captured_at,
            observed_at=captured_at,
            system_id=system,
            resource_id=rid,
            category=category,
            present=True,
            pii=pii,
            pii_fields_known=True,
            subject_ref=users[0] if users else None,
            links=links,
            source=f"{method}:{path.name}",
            extra={"document": path.name},
        )
        hints = STATE_HINTS.get(category)
        if category == "DATA_FLOW":
            obs.enabled = not any(k in lower for k in hints["disabled"]) and any(k in lower for k in hints["enabled"])
        elif category == "BACKUP_ARCHIVE":
            obs.restore_allowed = not any(k in lower for k in hints["blocked"])
        observations.append(obs)
    return observations, [], {"text": text, "method": method, "emails": doc_emails, "users": doc_users,
                             "resources": resources}


def alias_in_text(alias: str, text: str) -> bool:
    if not alias:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    return norm(alias) in norm(text)


__all__ = ["Grounder", "extract_document", "document_text", "alias_in_text"]
