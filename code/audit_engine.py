"""Per-case deletion audit: discovery -> state at audit_at -> policy -> steps.

Design rule: *let models describe, let deterministic code decide*.  Extractors
(and the optional LLM/VLM) only describe evidence.  Everything that determines
the answer -- which records belong to the subject, what state each record is in
at ``audit_at``, which policy applies, which evidence proves it, the step order
-- is computed here with plain, auditable code.

Temporal rules
--------------
* identity links may come from any evidence statement made at or before
  ``audit_at`` (a historical email in an old snapshot still identifies a copy);
* a record is *in scope* for a request only if some evidence observed it between
  ``requested_at`` and ``audit_at`` (older sightings describe a previous request
  of the same person) -- or the request's own deletion workflow touched it;
* the state of a record is the latest statement observed at or before
  ``audit_at`` in evidence captured at or before ``audit_at``; only such
  evidence may be cited.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from activity import ActivityIndex
from attribution import Attributed, JointAttributor, RequestSlot
from evidence_indexer import EvidenceIndex
from identity import Identity, IdentityResolver
from intent import IntentClassifier
from model import Activity, SYSTEM_DEFAULT_CATEGORY, Step, system_for_resource
from policy import PolicyEngine, RecordState
from steps import order_steps
from utils import parse_ts

LABELLED_SOURCES = ("text:", "json:lookup", "json:backup_restore_service", "json:privacy_treatment_service",
                    "json:support_storage_service", "log:")
DELETE_CATEGORIES = {"USER_PROFILE", "SECONDARY_IDENTIFIER", "SEARCH_INDEX_ENTRY", "CACHE_ENTRY",
                     "SUPPORT_ATTACHMENT", "USER_EXPORT", "MARKETING_PROFILE"}
ANALYTICS_CATEGORIES = {"ANALYTICS_PROFILE", "PRODUCT_TELEMETRY"}


@dataclass
class EngineConfig:
    """Feature switches (all on for submissions; toggled individually for ablations)."""

    historical_identifiers: bool = True   # use user_audit_log identities, not just the current snapshot
    request_window_scope: bool = True     # require an observation between requested_at and audit_at
    workflow_attribution: bool = True     # attribute identifier-less records via the deletion workflow
    snapshot_absence: bool = True         # prove absence from scoped snapshots that omit a record
    citable_evidence_only: bool = True    # state only from evidence captured at or before audit_at


@dataclass
class Request:
    case_id: str
    email: str
    requested_at: datetime
    reported_status: str
    reported_at: datetime
    audit_at: datetime
    request_text: str

    @classmethod
    def from_row(cls, row: dict) -> "Request":
        return cls(
            case_id=row["case_id"],
            email=row["email"],
            requested_at=parse_ts(row["requested_at"]),
            reported_status=row["reported_status"].strip().upper(),
            reported_at=parse_ts(row["reported_at"]),
            audit_at=parse_ts(row["audit_at"]),
            request_text=row["request_text"],
        )


def load_requests(path: Path) -> list[Request]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Request.from_row(r) for r in csv.DictReader(fh)]


@dataclass
class CaseResult:
    case_id: str
    status: str
    steps: list[Step]
    trace: dict = field(default_factory=dict)


@dataclass
class ResourceAudit:
    key: tuple
    via: list
    category: str
    state: RecordState
    decision: object


class AuditEngine:
    def __init__(self, dataset_dir: Path, index: EvidenceIndex, identities: IdentityResolver,
                 activity: ActivityIndex, policy: PolicyEngine, intent: IntentClassifier,
                 config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.dataset_dir = dataset_dir
        self.index = index
        self.identities = identities
        self.activity = activity
        self.policy = policy
        self.intent = intent
        self.primary: dict[str, Attributed] = {}
        self.narrative: dict[str, list[Attributed]] = {}
        self.anonymised: dict[str, Attributed] = {}
        self.notes: dict[str, list[str]] = {}
        self._owner_cache: dict[str, set[str]] = {}
        self._files_cache: dict[tuple, set[str]] = {}

    # ------------------------------------------------------------------ owners
    def owners_of_resource(self, resource_id: str) -> set[str]:
        """Subjects positively linked to a resource by identifiers anywhere in the evidence."""
        if resource_id in self._owner_cache:
            return self._owner_cache[resource_id]
        tokens: set[str] = set()
        for key in self.index.by_resource:
            if key[1] == resource_id:
                tokens |= self.index.link_tokens(key)
        for key, toks in self.index.catalog_links.items():
            if key[1] == resource_id:
                tokens |= toks
        owners: set[str] = set()
        for tok in tokens:
            owners |= self.identities.owner_of(tok)
        self._owner_cache[resource_id] = owners
        return owners

    def _files(self, key: tuple) -> set[str]:
        if key not in self._files_cache:
            self._files_cache[key] = {o.evidence_id for o in self.index.by_resource.get(key, [])}
        return self._files_cache[key]

    # ----------------------------------------------------------------- prepare
    def prepare(self, requests: list[Request]) -> None:
        """Joint workflow attribution over every deletion request in the input."""
        slots: list[RequestSlot] = []
        req_by_case = {}
        for r in requests:
            if not self.intent.classify(r.request_text).authorizes_deletion:
                continue
            ident = self.identities.resolve(r.email, r.audit_at)
            slots.append(RequestSlot(r.case_id, ident.user_id if ident else None, r.reported_at,
                                     r.reported_status, r.requested_at, r.audit_at))
            req_by_case[r.case_id] = r
        attributor = JointAttributor(self.activity, self.owners_of_resource)
        self.anonymised = {}
        if not self.config.workflow_attribution:
            self.primary, self.narrative, self.notes = {}, {}, {}
            return
        self.primary = attributor.primary_profiles(slots)
        self.narrative = attributor.narrative_records(slots)

        # First pass without identifier-less anonymised records: only requests whose
        # other records are already compliant (and that have no analytics profile of
        # their own) can own an anonymised profile completed by their workflow.
        first: dict[str, tuple] = {}
        for cid, r in req_by_case.items():
            res, audits = self._audit_core(r)
            first[cid] = (res.status, audits)

        def eligible(slot: RequestSlot, act: Activity) -> bool:
            status, audits = first[slot.case_id]
            if status != "COMPLETE":
                return False
            if any(a.category in ANALYTICS_CATEGORIES for a in audits):
                return False
            key = (act.target_system, act.resource_id)
            files = self._files(key)
            return any(files & self._files(a.key) for a in audits)

        self.anonymised = attributor.anonymised_records(slots, eligible)
        self.notes = attributor.notes

    # --------------------------------------------------------------- discovery
    def _discover(self, req: Request, ident: Identity) -> dict[tuple, set[str]]:
        audit_at = req.audit_at
        found: dict[tuple, set[str]] = defaultdict(set)
        catalog_ok = {(c.system_id, c.resource_id) for c in self.activity.catalog if c.event_time <= audit_at}
        tokens = ident.tokens if self.config.historical_identifiers else ident.current_tokens
        for tok in tokens:
            for key in self.index.by_token.get(tok, ()):
                ok = any(tok in o.links and o.observed_at <= audit_at for o in self.index.by_resource.get(key, []))
                if not ok and key in catalog_ok and tok in self.index.catalog_links.get(key, set()):
                    ok = True
                if ok:
                    found[key].add(f"identifier:{tok.split(':', 1)[0]}")
        # scope: the record was observed during this request's audit window
        in_scope = {}
        window_start = req.requested_at if self.config.request_window_scope else datetime.min
        for key, via in found.items():
            if any(window_start <= o.observed_at <= audit_at for o in self.index.by_resource.get(key, [])):
                in_scope[key] = set(via)
        # records naming an in-scope record as their source (search document -> account)
        for key in list(in_scope):
            for o in self.index.by_resource.get(key, []):
                src = o.extra.get("source_resource_id")
                if src and o.observed_at <= audit_at and system_for_resource(src):
                    in_scope.setdefault((system_for_resource(src), src), set()).add("source_reference")
        # the request's own deletion workflow
        att = self.primary.get(req.case_id)
        if att:
            in_scope.setdefault((att.system_id, att.resource_id), set()).add(
                "workflow:primary_profile" + (":ambiguous" if att.ambiguous else ""))
        for att in self.narrative.get(req.case_id, []):
            in_scope.setdefault((att.system_id, att.resource_id), set()).add("workflow:narrative")
        att = self.anonymised.get(req.case_id)
        if att:
            in_scope.setdefault((att.system_id, att.resource_id), set()).add(
                "workflow:completed_action" + (":ambiguous" if att.ambiguous else ""))
        return in_scope

    # ---------------------------------------------------------------- category
    def _category(self, key: tuple, audit_at: datetime) -> str:
        system, rid = key
        if system == "integration_service":
            return "DATA_FLOW"
        obs = [o for o in self.index.by_resource.get(key, []) if o.observed_at <= audit_at]
        labelled = [o.category for o in obs if o.category and o.source.startswith(LABELLED_SOURCES)
                    and o.category != "DATA_FLOW"]
        if labelled:
            return labelled[-1]
        cats = [o.category for o in obs if o.category]
        if "SUPPORT_ATTACHMENT" in cats:
            return "SUPPORT_ATTACHMENT"
        default = SYSTEM_DEFAULT_CATEGORY.get(system, "USER_PROFILE")
        intent = self.activity.treatment_intent(rid, before=audit_at)
        if intent == "DELETE" and default in ANALYTICS_CATEGORIES:
            return "SECONDARY_IDENTIFIER"   # identifier-keyed copy the workflow deletes outright
        if intent == "DELETE" and default == "SUPPORT_MESSAGE":
            return "SUPPORT_ATTACHMENT"
        return cats[-1] if cats else default

    # ------------------------------------------------------------------- state
    def _state(self, key: tuple, category: str, req: Request, scope_keys: set) -> RecordState:
        system, rid = key
        audit_at = req.audit_at
        obs = [o for o in self.index.by_resource.get(key, [])
               if o.observed_at <= audit_at
               and (o.file_captured_at <= audit_at or not self.config.citable_evidence_only)]
        obs.sort(key=lambda o: (o.observed_at, o.file_captured_at, o.evidence_id))

        present: Optional[bool] = None
        present_time: Optional[datetime] = None
        present_ev = ""
        for o in obs:
            if o.present is None:
                continue
            if present_time is None or o.observed_at > present_time:
                present, present_time, present_ev = o.present, o.observed_at, o.evidence_id
            elif o.observed_at == present_time and o.present and not present:
                # same capture, several tables / replicas: any surviving copy counts
                present, present_ev = True, o.evidence_id

        action, _ = self.policy.action_for(category)
        if action == "DELETE" and self.config.snapshot_absence:
            cov_best = None
            for cov in self.index.coverage:
                if cov.system_id != system or rid in cov.resource_ids:
                    continue
                if cov.captured_at > audit_at or cov.file_captured_at > audit_at:
                    continue
                if present_time is not None:
                    if cov.captured_at <= present_time:
                        continue
                    if not any(cov.evidence_id in self._files(k) for k in scope_keys if k != key):
                        continue
                if cov_best is None or (cov.captured_at, cov.evidence_id) > (cov_best.captured_at, cov_best.evidence_id):
                    cov_best = cov
            if cov_best is not None:
                present, present_time, present_ev = False, cov_best.captured_at, cov_best.evidence_id

        exposed: Optional[bool] = None
        exposed_ev = ""
        analytics_like = category in ANALYTICS_CATEGORIES
        for o in obs:
            if not o.pii_fields_known or o.present is False:
                continue
            has = o.has_direct_identifiers() or (analytics_like and bool(o.subject_ref))
            exposed, exposed_ev = has, o.evidence_id
        enabled, enabled_ev = None, ""
        restore, restore_ev = None, ""
        for o in obs:
            if o.enabled is not None:
                enabled, enabled_ev = o.enabled, o.evidence_id
            if o.restore_allowed is not None:
                restore, restore_ev = o.restore_allowed, o.evidence_id

        state = RecordState(system, rid, category, present, exposed, enabled, restore)
        if category == "DATA_FLOW":
            state.proof_evidence = state.violation_evidence = enabled_ev or present_ev
        elif category == "BACKUP_ARCHIVE":
            state.proof_evidence = state.violation_evidence = restore_ev or present_ev
        elif action in {"REDACT", "ANONYMIZE"}:
            state.proof_evidence = state.violation_evidence = exposed_ev or present_ev
            if present is False:
                state.proof_evidence = present_ev
        else:
            state.proof_evidence = state.violation_evidence = present_ev
        return state

    # -------------------------------------------------------------------- audit
    def _audit_core(self, req: Request):
        ident = self.identities.resolve(req.email, req.audit_at)
        if ident is None:
            return CaseResult(req.case_id, "COMPLETE", [], {"identity": "unresolved"}), []
        scope = self._discover(req, ident)
        scope_keys = set(scope)
        audits: list[ResourceAudit] = []
        for key in sorted(scope):
            category = self._category(key, req.audit_at)
            state = self._state(key, category, req, scope_keys)
            audits.append(ResourceAudit(key, sorted(scope[key]), category, state, self.policy.evaluate(state)))

        incomplete = any(not a.decision.compliant for a in audits)
        status = "INCOMPLETE" if incomplete else "COMPLETE"
        steps: list[Step] = []
        for a in audits:
            system, rid = a.key
            d = a.decision
            if not d.compliant:
                steps.append(Step("FINDING", system, rid, d.finding, a.state.violation_evidence))
                steps.append(Step("REMEDIATION", system, rid, d.remediation, ""))
            steps.append(Step("VERIFICATION", system, rid, d.verification, "" if incomplete else a.state.proof_evidence))
        trace = {
            "identity": ident.describe(),
            "resources": [
                {
                    "system_id": a.key[0], "resource_id": a.key[1], "via": a.via, "category": a.category,
                    "present": a.state.present, "exposed": a.state.exposed_identifiers, "enabled": a.state.enabled,
                    "restore_allowed": a.state.restore_allowed, "action": a.decision.action,
                    "compliant": a.decision.compliant, "proof": a.state.proof_evidence,
                }
                for a in audits
            ],
        }
        return CaseResult(req.case_id, status, order_steps(steps), trace), audits

    def audit(self, req: Request) -> CaseResult:
        intent = self.intent.classify(req.request_text)
        if not intent.authorizes_deletion:
            return CaseResult(req.case_id, "NOT_APPLICABLE", [],
                              {"intent": intent.label, "intent_reason": intent.reason, "intent_source": intent.source})
        result, _ = self._audit_core(req)
        result.trace.update({"intent": intent.label, "intent_reason": intent.reason, "intent_source": intent.source})
        if self.notes.get(req.case_id):
            result.trace["attribution_notes"] = list(self.notes[req.case_id])
        return result


__all__ = ["AuditEngine", "EngineConfig", "Request", "CaseResult", "load_requests"]
