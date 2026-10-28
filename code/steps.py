"""Canonical step ordering and serialisation for ``steps_json``.

Order: phase (FINDING -> REMEDIATION -> VERIFICATION), then the operation order
from the problem statement, then ``system_id`` and ``resource_id`` ascending.
Step numbers are assigned after sorting, starting at 1.  Serialisation uses a
fixed key order and compact separators so output is byte-for-byte reproducible.
"""

from __future__ import annotations

import json

from model import OPERATION_ORDER, OPERATION_PHASE, OPERATION_STATE, PHASES, Step

STEP_KEYS = ("step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id")


def order_steps(steps: list[Step]) -> list[Step]:
    unique = {}
    for s in steps:
        unique.setdefault((s.phase, s.system_id, s.resource_id, s.operation), s)
    return sorted(unique.values(), key=lambda s: s.sort_key())


def to_json_list(steps: list[Step]) -> list[dict]:
    out = []
    for i, s in enumerate(order_steps(steps), start=1):
        out.append({
            "step_no": i,
            "phase": s.phase,
            "system_id": s.system_id,
            "resource_id": s.resource_id,
            "operation": s.operation,
            "state": OPERATION_STATE[s.operation],
            "evidence_id": s.evidence_id or "",
        })
    return out


def serialize(steps: list[Step]) -> str:
    return json.dumps(to_json_list(steps), ensure_ascii=False, separators=(",", ":"))


def validate_steps(items: list[dict], status: str) -> list[str]:
    """Return contract violations for one row (empty list = valid)."""
    errors = []
    if status == "NOT_APPLICABLE" and items:
        errors.append("NOT_APPLICABLE must have no steps")
    for i, s in enumerate(items, start=1):
        if set(s) != set(STEP_KEYS):
            errors.append(f"step {i}: keys {sorted(s)}")
        if s.get("step_no") != i:
            errors.append(f"step {i}: step_no {s.get('step_no')}")
        op = s.get("operation")
        if op not in OPERATION_STATE:
            errors.append(f"step {i}: unknown operation {op}")
            continue
        if s.get("phase") != OPERATION_PHASE[op]:
            errors.append(f"step {i}: phase {s.get('phase')} for {op}")
        if s.get("state") != OPERATION_STATE[op]:
            errors.append(f"step {i}: state {s.get('state')} for {op}")
        if status == "COMPLETE" and s.get("phase") != "VERIFICATION":
            errors.append(f"step {i}: COMPLETE rows may only verify")
        if status == "COMPLETE" and not s.get("evidence_id"):
            errors.append(f"step {i}: COMPLETE verification without evidence")
        if status == "INCOMPLETE" and s.get("phase") in {"REMEDIATION", "VERIFICATION"} and s.get("evidence_id"):
            errors.append(f"step {i}: future step cites evidence")
        if s.get("phase") == "FINDING" and not s.get("evidence_id"):
            errors.append(f"step {i}: finding without evidence")
    keys = [(PHASES.index(s["phase"]), OPERATION_ORDER[s["phase"]].index(s["operation"]), s["system_id"],
             s["resource_id"]) for s in items if s.get("operation") in OPERATION_STATE]
    if keys != sorted(keys):
        errors.append("steps not in canonical order")
    return errors


__all__ = ["order_steps", "to_json_list", "serialize", "validate_steps", "STEP_KEYS"]
