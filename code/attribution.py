"""Joint workflow attribution for identifier-less records.

A deleted account that now answers 404, or an analytics profile that has been
anonymised, no longer carries any identifier, so evidence alone cannot say whose
it was.  The deletion workflow, however, leaves a timed trail in the activity
stream: each request's primary profile is submitted 40-80 minutes before the
request's ``reported_at``, and completed policy actions (e.g. an analytics
anonymisation) land in the 36 minutes before it.

Many requests share a ``reported_at`` slot and the stream also contains decoy
submissions, so attribution is solved *jointly* over every request in the
input -- a constrained matching problem, never a lookup of answers:

* an event whose resource is positively linked by identifiers to a subject
  belongs to that subject (and is excluded for everyone else);
* the primary-profile event's logged result must agree with the request's
  reported status (the workflow logs SUCCESS for COMPLETE, PARTIAL for
  PARTIALLY_COMPLETE and FAILED for NOT_STARTED);
* each request has exactly one primary profile and each event one owner;
* unique candidates are propagated to a fixed point; residual ties are broken
  deterministically (time order <-> request order) and flagged ``ambiguous``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from activity import (
    ActivityIndex, PRIMARY_PROFILE_RE, TARGET_RESOURCE_RE, STATUS_RESULT,
    PRIMARY_WINDOW, COMPLETED_ACTION_WINDOW, NARRATIVE_WINDOW,
)
from model import Activity, system_for_resource


@dataclass
class Attributed:
    system_id: str
    resource_id: str
    kind: str
    activity_id: str
    ambiguous: bool = False
    candidates: int = 1


@dataclass
class RequestSlot:
    case_id: str
    user_id: Optional[str]
    reported_at: datetime
    reported_status: str
    requested_at: datetime
    audit_at: datetime


def _offset(a: Activity, slot: RequestSlot) -> float:
    return (a.event_time - slot.reported_at).total_seconds() / 60.0


def _propagate(candidates: dict[str, set[str]], order_key_case, order_key_event, notes, label):
    """Unique-candidate propagation for a one-to-one assignment, then ordered tie-breaking.

    Only request-side uniqueness is propagated (a request with a single live candidate
    takes it).  Event-side uniqueness ("only one request could own this event") is *not*
    used: the stream contains decoy submissions, so an event need not belong to anyone.
    Every loop runs in a fixed order, so the result never depends on hash seeds.
    """
    assigned: dict[str, str] = {}
    taken: set[str] = set()
    order = sorted(candidates, key=order_key_case)
    changed = True
    while changed:
        changed = False
        for cid in order:
            if cid in assigned:
                continue
            live = candidates[cid] - taken
            if len(live) == 1:
                ev = next(iter(live))
                assigned[cid] = ev
                taken.add(ev)
                changed = True
    ambiguous: set[str] = set()
    for cid in sorted((c for c in candidates if c not in assigned), key=order_key_case):
        live = sorted(candidates[cid] - taken, key=order_key_event)
        if not live:
            continue
        assigned[cid] = live[0]
        taken.add(live[0])
        ambiguous.add(cid)
        notes[cid].append(f"{label}: tie among {len(live)} candidates")
    return assigned, ambiguous


class JointAttributor:
    def __init__(self, activity: ActivityIndex, owners_of_resource: Callable[[str], set]):
        self.activity = activity
        self.owners = owners_of_resource
        self.notes: dict[str, list[str]] = defaultdict(list)

    def primary_profiles(self, slots: list[RequestSlot]) -> dict[str, Attributed]:
        acts = [a for a in self.activity.activities if a.source_system == "deletion_worker"]
        events = {a.activity_id: a for a in acts if PRIMARY_PROFILE_RE.search(a.details)}
        # hand-written workflows report the profile deletion as a narrative "Target resource" event
        for a in acts:
            if TARGET_RESOURCE_RE.search(a.details) and a.target_system == "account_service":
                events[a.activity_id] = a
        slot_by_case = {s.case_id: s for s in slots}
        candidates: dict[str, set[str]] = {}
        for s in slots:
            want = STATUS_RESULT.get(s.reported_status)
            cands: set[str] = set()
            direct: set[str] = set()
            for a in events.values():
                narrative = bool(TARGET_RESOURCE_RE.search(a.details))
                window = NARRATIVE_WINDOW if narrative else PRIMARY_WINDOW
                off = _offset(a, s)
                if not (window[0] <= off <= window[1]):
                    continue
                owners = self.owners(a.resource_id)
                if owners:
                    if s.user_id in owners:
                        direct.add(a.activity_id)
                    continue
                if not narrative and want and a.reported_result != want:
                    continue
                cands.add(a.activity_id)
            candidates[s.case_id] = direct or cands
        assigned, ambiguous = _propagate(
            candidates,
            order_key_case=lambda c: (slot_by_case[c].reported_at, c),
            order_key_event=lambda e: (events[e].event_time, e),
            notes=self.notes, label="primary profile",
        )
        return {
            cid: Attributed("account_service", events[ev].resource_id, "primary_profile", ev,
                            ambiguous=cid in ambiguous, candidates=len(candidates[cid]))
            for cid, ev in assigned.items()
        }

    def narrative_records(self, slots: list[RequestSlot]) -> dict[str, list[Attributed]]:
        """Non-account 'Target resource' events of hand-written workflows (unique window match only)."""
        out: dict[str, list[Attributed]] = defaultdict(list)
        for a in self.activity.activities:
            if a.source_system != "deletion_worker" or a.target_system == "account_service":
                continue
            if not TARGET_RESOURCE_RE.search(a.details):
                continue
            owners = self.owners(a.resource_id)
            cands = [s for s in slots if NARRATIVE_WINDOW[0] <= _offset(a, s) <= NARRATIVE_WINDOW[1]
                     and (not owners or s.user_id in owners)]
            if len(cands) == 1:
                out[cands[0].case_id].append(
                    Attributed(a.target_system or system_for_resource(a.resource_id), a.resource_id, "narrative",
                               a.activity_id))
        return out

    def anonymised_records(self, slots: list[RequestSlot], eligible: Callable[[RequestSlot, Activity], bool]
                           ) -> dict[str, Attributed]:
        """Assign identifier-less 'Policy action completed' anonymisations to eligible requests."""
        events = {
            a.activity_id: a for a in self.activity.activities
            if a.source_system == "deletion_worker" and a.event_type == "ANONYMIZE_RECORD"
            and a.details.startswith("Policy action completed") and not self.owners(a.resource_id)
        }
        slot_by_case = {s.case_id: s for s in slots}
        candidates: dict[str, set[str]] = {}
        for s in slots:
            cands = set()
            for a in events.values():
                off = _offset(a, s)
                if COMPLETED_ACTION_WINDOW[0] <= off <= COMPLETED_ACTION_WINDOW[1] and eligible(s, a):
                    cands.add(a.activity_id)
            if cands:
                candidates[s.case_id] = cands
        assigned, ambiguous = _propagate(
            candidates,
            order_key_case=lambda c: (slot_by_case[c].reported_at, c),
            order_key_event=lambda e: (events[e].event_time, e),
            notes=self.notes, label="anonymised analytics",
        )
        return {
            cid: Attributed(events[ev].target_system, events[ev].resource_id, "completed_action", ev,
                            ambiguous=cid in ambiguous, candidates=len(candidates[cid]))
            for cid, ev in assigned.items()
        }


__all__ = ["JointAttributor", "RequestSlot", "Attributed"]
