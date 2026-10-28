"""Build the once-per-run evidence index.

Reads ``evidence_files.csv``, dispatches every file to its extractor, grounds
OCR/PDF identifiers against identifiers known from structured data, applies the
data-catalog links from the activity stream, and exposes lookups by resource
and by identifier token.  Nothing here is case-specific: the manifest does not
say which case a file belongs to, so evidence is connected to subjects purely
through the identifiers it contains.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from activity import ActivityIndex
from extractors.documents import Grounder, alias_in_text, extract_document
from extractors.json_records import extract_json, extract_text
from extractors.logs import extract_log
from extractors.sql_snapshots import extract_sql
from identity import IdentityResolver
from model import Observation, SnapshotCoverage, SYSTEM_DEFAULT_CATEGORY, system_for_resource
from utils import parse_ts


@dataclass
class EvidenceFile:
    file_id: str
    path: Path
    origin: str
    captured_at: datetime


@dataclass
class EvidenceIndex:
    files: dict[str, EvidenceFile] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    coverage: list[SnapshotCoverage] = field(default_factory=list)
    by_resource: dict = field(default_factory=lambda: defaultdict(list))
    by_token: dict = field(default_factory=lambda: defaultdict(set))
    documents: dict = field(default_factory=dict)        # file_id -> extracted text metadata
    catalog_links: dict = field(default_factory=lambda: defaultdict(set))  # (system, rid) -> tokens
    stats: dict = field(default_factory=dict)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)
        self.by_resource[obs.key].append(obs)
        for tok in obs.links:
            self.by_token[tok].add(obs.key)

    def link_tokens(self, key) -> set[str]:
        toks = set(self.catalog_links.get(key, set()))
        for obs in self.by_resource.get(key, []):
            toks |= obs.links
        return toks


def build_index(dataset_dir: Path, activity: ActivityIndex, identities: IdentityResolver,
                verbose: bool = False, vlm=None) -> EvidenceIndex:
    t0 = time.time()
    idx = EvidenceIndex()
    with open(dataset_dir / "evidence_files.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            idx.files[row["file_id"]] = EvidenceFile(
                file_id=row["file_id"],
                path=dataset_dir / row["file_path"],
                origin=row["origin"],
                captured_at=parse_ts(row["captured_at"]),
            )

    structured = {".sql": extract_sql, ".json": extract_json, ".log": extract_log, ".txt": extract_text}
    documents = []
    per_type = defaultdict(int)
    for ef in idx.files.values():
        suffix = ef.path.suffix.lower()
        if suffix in structured:
            obs, cov = structured[suffix](ef.path, ef.file_id, ef.captured_at)
            for o in obs:
                idx.add(o)
            idx.coverage.extend(cov)
            per_type[suffix] += 1
        elif suffix in {".pdf", ".png", ".jpg", ".jpeg"}:
            documents.append(ef)

    # Ground OCR/PDF identifiers against what structured data already knows.
    known_ids = {o.resource_id for o in idx.observations} | activity.resource_ids
    known_ids |= {c.resource_id for c in activity.catalog}
    known_emails = set(identities.email_index)
    known_emails |= {t.split(":", 1)[1] for t in idx.by_token if t.startswith("email:")}
    known_emails |= {c.email for c in activity.catalog if c.email}
    grounder = Grounder(resource_ids=known_ids, emails=known_emails, users=set(identities.users))
    for ef in documents:
        obs, cov, meta = extract_document(ef.path, ef.file_id, ef.captured_at, grounder, vlm=vlm)
        idx.documents[ef.file_id] = meta
        for o in obs:
            idx.add(o)
        per_type[ef.path.suffix.lower()] += 1

    _apply_catalog(idx, activity)
    idx.stats = {
        "files": len(idx.files),
        "files_by_type": dict(per_type),
        "observations": len(idx.observations),
        "resources": len(idx.by_resource),
        "coverage_entries": len(idx.coverage),
        "catalog_links": len(activity.catalog),
        "build_seconds": round(time.time() - t0, 2),
    }
    if verbose:
        print("[index]", idx.stats)
    return idx


def _apply_catalog(idx: EvidenceIndex, activity: ActivityIndex) -> None:
    """Attach data-catalog identity links and materialise the documents they describe."""
    for link in activity.catalog:
        system = link.system_id or system_for_resource(link.resource_id)
        key = (system, link.resource_id)
        tokens = set()
        if link.email:
            tokens.add(f"email:{link.email}")
        if link.user_id:
            tokens.add(f"user:{link.user_id}")
        idx.catalog_links[key] |= tokens
        for tok in tokens:
            idx.by_token[tok].add(key)

        # Which captured document does this catalog entry describe?  The one that
        # mentions the alias / resource id and was captured shortly before the link.
        best = None
        for fid, meta in idx.documents.items():
            ef = idx.files[fid]
            if ef.captured_at > link.event_time or link.event_time - ef.captured_at > timedelta(hours=6):
                continue
            text = meta.get("text", "")
            if alias_in_text(link.alias, text) or alias_in_text(link.resource_id, text):
                gap = link.event_time - ef.captured_at
                if best is None or gap < best[0]:
                    best = (gap, fid)
        if best is None:
            continue
        fid = best[1]
        ef = idx.files[fid]
        existing = [o for o in idx.by_resource.get(key, []) if o.evidence_id == fid]
        if existing:
            for o in existing:
                o.links |= tokens
            continue
        meta = idx.documents[fid]
        text_lower = meta.get("text", "").lower()
        category = SYSTEM_DEFAULT_CATEGORY.get(system)
        pii = {}
        visible = [e for e in meta.get("emails", []) if e == link.email]
        if visible and category not in {"SECURITY_AUDIT_LOG", "ACCOUNT_TOMBSTONE"}:
            pii["email"] = visible[0]
        if category == "ACCOUNT_TOMBSTONE" and link.email and link.email in text_lower:
            pii["email"] = link.email
        obs = Observation(
            evidence_id=fid,
            file_captured_at=ef.captured_at,
            observed_at=ef.captured_at,
            system_id=system,
            resource_id=link.resource_id,
            category=category,
            present=True,
            pii=pii,
            pii_fields_known=True,
            subject_ref=link.user_id,
            links=set(tokens),
            source=f"catalog+{meta.get('method')}:{ef.path.name}",
            extra={"document": ef.path.name, "catalog_activity": link.activity_id, "alias": link.alias},
        )
        idx.add(obs)


__all__ = ["EvidenceIndex", "EvidenceFile", "build_index"]
