"""Deletion policy (``deletion_rules.csv``) and per-record compliance evaluation.

``PolicyEngine.evaluate`` receives the reconstructed state of one record at
``audit_at`` and returns the ordered findings / remediation / verification
operations the policy demands.  The mapping is deliberately tiny and
table-driven so it can be read next to the rules file:

=====================  =================  ==========================================
required action        compliant state    violation -> finding / remediation
=====================  =================  ==========================================
DELETE                 ABSENT             REPORT_REMAINING_RECORD / DELETE_RECORD
REDACT                 REDACTED           REPORT_EXPOSED_PII / REDACT_RECORD
ANONYMIZE              ANONYMIZED         REPORT_EXPOSED_PII / ANONYMIZE_RECORD
RETAIN                 PRESENT            (must stay; never deleted)
RETAIN (backup)        restore BLOCKED    REPORT_RESTORE_RISK / BLOCK_RESTORE
data flow              DISABLED           REPORT_ACTIVE_DATA_FLOW / DISABLE_DATA_FLOW
=====================  =================  ==========================================
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VERIFY_FOR_ACTION = {
    "DELETE": "VERIFY_RECORD_ABSENT",
    "REDACT": "VERIFY_RECORD_REDACTED",
    "ANONYMIZE": "VERIFY_RECORD_ANONYMIZED",
    "RETAIN": "VERIFY_RECORD_PRESENT",
    "BLOCK_RESTORE": "VERIFY_RESTORE_BLOCKED",
    "DISABLE_DATA_FLOW": "VERIFY_DATA_FLOW_DISABLED",
}

FINDING_FOR_ACTION = {
    "DELETE": "REPORT_REMAINING_RECORD",
    "REDACT": "REPORT_EXPOSED_PII",
    "ANONYMIZE": "REPORT_EXPOSED_PII",
    "BLOCK_RESTORE": "REPORT_RESTORE_RISK",
    "DISABLE_DATA_FLOW": "REPORT_ACTIVE_DATA_FLOW",
}

REMEDIATION_FOR_ACTION = {
    "DELETE": "DELETE_RECORD",
    "REDACT": "REDACT_RECORD",
    "ANONYMIZE": "ANONYMIZE_RECORD",
    "BLOCK_RESTORE": "BLOCK_RESTORE",
    "DISABLE_DATA_FLOW": "DISABLE_DATA_FLOW",
}


@dataclass
class Rule:
    rule_id: str
    category: str
    condition: str
    action: str


@dataclass
class RecordState:
    """Reconstructed state of one record at audit time."""

    system_id: str
    resource_id: str
    category: str
    present: Optional[bool]              # None = unknown
    exposed_identifiers: Optional[bool]  # direct identifiers (or, for analytics, any link) still present
    enabled: Optional[bool] = None
    restore_allowed: Optional[bool] = None
    proof_evidence: str = ""             # evidence proving the current state
    violation_evidence: str = ""         # evidence proving a violation


@dataclass
class Decision:
    action: str                 # required action after mapping (DELETE/REDACT/ANONYMIZE/RETAIN/BLOCK_RESTORE/DISABLE_DATA_FLOW)
    compliant: bool
    finding: Optional[str]
    remediation: Optional[str]
    verification: str
    rule_id: str


class PolicyEngine:
    def __init__(self, dataset_dir: Path):
        self.rules: dict[str, Rule] = {}
        with open(dataset_dir / "deletion_rules.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.rules[row["data_category"]] = Rule(
                    row["rule_id"], row["data_category"], row["condition"], row["required_action"]
                )

    def action_for(self, category: str) -> tuple[str, str]:
        """Return (required action, rule id) for a data category."""
        if category == "DATA_FLOW":
            return "DISABLE_DATA_FLOW", "active-flow"
        rule = self.rules.get(category)
        if rule is None:
            return "DELETE", "default"
        if category == "BACKUP_ARCHIVE":
            return "BLOCK_RESTORE", rule.rule_id
        return rule.action, rule.rule_id

    def evaluate(self, state: RecordState, action_override: Optional[str] = None) -> Decision:
        action, rule_id = self.action_for(state.category)
        if action_override:
            action = action_override
        verification = VERIFY_FOR_ACTION[action]

        if action == "DELETE":
            compliant = state.present is not True
        elif action in {"REDACT", "ANONYMIZE"}:
            if state.present is False:
                # the copy is gone entirely: nothing left to expose
                return Decision(action, True, None, None, "VERIFY_RECORD_ABSENT", rule_id)
            compliant = not state.exposed_identifiers
        elif action == "RETAIN":
            compliant = True
        elif action == "BLOCK_RESTORE":
            compliant = state.restore_allowed is not True
        elif action == "DISABLE_DATA_FLOW":
            compliant = state.enabled is not True
        else:
            compliant = True

        if compliant:
            return Decision(action, True, None, None, verification, rule_id)
        return Decision(
            action, False, FINDING_FOR_ACTION.get(action), REMEDIATION_FOR_ACTION.get(action), verification, rule_id
        )


__all__ = ["PolicyEngine", "RecordState", "Decision", "Rule"]
