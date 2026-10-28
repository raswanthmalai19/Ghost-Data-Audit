"""Temporal identity resolution.

A request email may be the subject's current address or one they used before an
identity change.  ``IdentityResolver`` merges the current snapshot
(``user_identities.csv``) with the change history (``user_audit_log.csv``) and
returns every identifier the subject has ever used, each tagged with its
provenance (``current`` or the ``audit_id`` that recorded it).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import email_hash, norm_email, norm_hash, norm_name, norm_phone, parse_ts, token


@dataclass
class Identity:
    user_id: str
    name: str
    emails: dict[str, str] = field(default_factory=dict)   # email -> provenance
    hashes: dict[str, str] = field(default_factory=dict)   # sha256 -> provenance
    phones: dict[str, str] = field(default_factory=dict)   # digits -> provenance
    names: set[str] = field(default_factory=set)
    matched_via: str = "current"                            # how the request email matched
    audit_ids: list[str] = field(default_factory=list)      # identity-change audit rows

    @property
    def tokens(self) -> set[str]:
        out = {f"user:{self.user_id}"}
        out.update(f"email:{e}" for e in self.emails)
        out.update(f"hash:{h}" for h in self.hashes)
        out.update(f"phone:{p}" for p in self.phones)
        return out

    @property
    def current_tokens(self) -> set[str]:
        """Only identifiers from the current snapshot (used for the historical-identity ablation)."""
        out = {f"user:{self.user_id}"}
        out.update(f"email:{e}" for e, p in self.emails.items() if p == "current")
        out.update(f"hash:{h}" for h, p in self.hashes.items() if p == "current")
        out.update(f"phone:{p}" for p, prov in self.phones.items() if prov == "current")
        return out

    @property
    def name_tokens(self) -> set[str]:
        return {f"name:{n}" for n in self.names}

    def describe(self) -> dict:
        return {
            "user_id": self.user_id,
            "matched_via": self.matched_via,
            "emails": self.emails,
            "phones": self.phones,
            "hash_count": len(self.hashes),
            "identity_change_audit_ids": self.audit_ids,
        }


class IdentityResolver:
    def __init__(self, dataset_dir: Path):
        self.users: dict[str, dict] = {}
        with open(dataset_dir / "user_identities.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.users[row["user_id"].lower()] = row
        self.changes: dict[str, list[dict]] = {}
        with open(dataset_dir / "user_audit_log.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.changes.setdefault(row["user_id"].lower(), []).append(row)
        for rows in self.changes.values():
            rows.sort(key=lambda r: (r["event_time"], r["audit_id"]))

        # email -> [(user_id, provenance, valid_from, valid_to)]
        self.email_index: dict[str, list[tuple[str, str, Optional[datetime], Optional[datetime]]]] = {}
        for uid, row in self.users.items():
            for email, prov, start, end in self._email_timeline(uid):
                self.email_index.setdefault(email, []).append((uid, prov, start, end))

    def _email_timeline(self, uid: str):
        """Yield (email, provenance, valid_from, valid_to) for every address the user held."""
        current = norm_email(self.users[uid]["email"])
        changes = [c for c in self.changes.get(uid, []) if c["field_name"] == "email"]
        start = None
        for c in changes:
            old = norm_email(c["old_value"])
            when = parse_ts(c["event_time"])
            if old:
                yield old, c["audit_id"], start, when
            start = when
        if current:
            yield current, "current", start, None

    def resolve(self, email: str, at: Optional[datetime] = None) -> Optional[Identity]:
        key = norm_email(email)
        if not key or key not in self.email_index:
            return None
        candidates = self.email_index[key]
        if len(candidates) > 1 and at is not None:
            # Prefer the holder of the address at the audit time.
            live = [c for c in candidates if (c[2] is None or c[2] <= at) and (c[3] is None or at < c[3])]
            if live:
                candidates = live
        uid, provenance, _, _ = sorted(candidates, key=lambda c: (c[1] != "current", c[0]))[0]
        return self.build(uid, matched_via=provenance)

    def build(self, uid: str, matched_via: str = "current") -> Identity:
        row = self.users[uid]
        ident = Identity(user_id=uid, name=row["name"], matched_via=matched_via)
        ident.emails[norm_email(row["email"])] = "current"
        h = norm_hash(row["email_hash"])
        if h:
            ident.hashes[h] = "current"
        p = norm_phone(row["phone"])
        if p:
            ident.phones[p] = "current"
        n = norm_name(row["name"])
        if n:
            ident.names.add(n)
        for c in self.changes.get(uid, []):
            ident.audit_ids.append(c["audit_id"])
            for value in (c["old_value"], c["new_value"]):
                if c["field_name"] == "email":
                    e = norm_email(value)
                    if e:
                        ident.emails.setdefault(e, c["audit_id"])
                elif c["field_name"] == "email_hash":
                    hv = norm_hash(value)
                    if hv:
                        ident.hashes.setdefault(hv, c["audit_id"])
                elif c["field_name"] == "phone":
                    pv = norm_phone(value)
                    if pv:
                        ident.phones.setdefault(pv, c["audit_id"])
                elif c["field_name"] == "name":
                    nv = norm_name(value)
                    if nv:
                        ident.names.add(nv)
        # Hashes of every known address, so systems that only store a hash still link.
        for e, prov in list(ident.emails.items()):
            ident.hashes.setdefault(email_hash(e), prov)
        return ident

    def owner_of(self, tok: str) -> set[str]:
        """Return the user ids that own an identifier token (used to exclude other subjects)."""
        kind, _, value = tok.partition(":")
        owners: set[str] = set()
        if kind == "user":
            if value in self.users:
                owners.add(value)
        elif kind == "email":
            owners.update(c[0] for c in self.email_index.get(value, []))
        else:
            for uid in self._token_owner_index().get(tok, ()):  # hashes / phones
                owners.add(uid)
        return owners

    _owner_cache: Optional[dict[str, set[str]]] = None

    def _token_owner_index(self) -> dict[str, set[str]]:
        if self._owner_cache is None:
            idx: dict[str, set[str]] = {}
            for uid in self.users:
                ident = self.build(uid)
                for tok in ident.tokens:
                    if tok.startswith(("hash:", "phone:")):
                        idx.setdefault(tok, set()).add(uid)
            self._owner_cache = idx
        return self._owner_cache


def identity_tokens_for(resolver: IdentityResolver, uid: str) -> set[str]:
    return resolver.build(uid).tokens


__all__ = ["Identity", "IdentityResolver", "identity_tokens_for", "token"]
