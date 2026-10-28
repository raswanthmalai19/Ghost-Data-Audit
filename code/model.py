"""Core data model shared by the extractors, the audit engine and the step writer.

Every evidence format (SQL snapshot, JSON export, log line, handover text, PDF,
screenshot) is normalised into ``Observation`` records.  An observation states
what one participant-visible evidence item says about one resource at one point
in time, plus the identifiers that link it to a data subject.  Downstream code
never touches raw files again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Canonical vocabularies
# ---------------------------------------------------------------------------

PHASES = ("FINDING", "REMEDIATION", "VERIFICATION")

OPERATION_ORDER = {
    "FINDING": [
        "REPORT_REMAINING_RECORD",
        "REPORT_EXPOSED_PII",
        "REPORT_ACTIVE_DATA_FLOW",
        "REPORT_RESTORE_RISK",
    ],
    "REMEDIATION": [
        "DISABLE_DATA_FLOW",
        "BLOCK_RESTORE",
        "DELETE_RECORD",
        "REDACT_RECORD",
        "ANONYMIZE_RECORD",
    ],
    "VERIFICATION": [
        "VERIFY_RECORD_ABSENT",
        "VERIFY_RECORD_PRESENT",
        "VERIFY_RECORD_REDACTED",
        "VERIFY_RECORD_ANONYMIZED",
        "VERIFY_DATA_FLOW_DISABLED",
        "VERIFY_RESTORE_BLOCKED",
    ],
}

OPERATION_STATE = {
    "REPORT_REMAINING_RECORD": "PRESENT",
    "REPORT_EXPOSED_PII": "PRESENT",
    "REPORT_ACTIVE_DATA_FLOW": "ACTIVE",
    "REPORT_RESTORE_RISK": "ACTIVE",
    "DISABLE_DATA_FLOW": "DISABLED",
    "BLOCK_RESTORE": "BLOCKED",
    "DELETE_RECORD": "ABSENT",
    "REDACT_RECORD": "REDACTED",
    "ANONYMIZE_RECORD": "ANONYMIZED",
    "VERIFY_RECORD_ABSENT": "ABSENT",
    "VERIFY_RECORD_PRESENT": "PRESENT",
    "VERIFY_RECORD_REDACTED": "REDACTED",
    "VERIFY_RECORD_ANONYMIZED": "ANONYMIZED",
    "VERIFY_DATA_FLOW_DISABLED": "DISABLED",
    "VERIFY_RESTORE_BLOCKED": "BLOCKED",
}

OPERATION_PHASE = {op: phase for phase, ops in OPERATION_ORDER.items() for op in ops}

# Resource-id prefix -> owning system.  Used when evidence names a resource
# without saying which service holds it (e.g. OCR text, free-form documents).
PREFIX_SYSTEM = {
    "account": "account_service",
    "search": "search_service",
    "support": "support_platform",
    "export": "export_service",
    "analytics": "analytics_platform",
    "backup": "backup_service",
    "data_flow": "integration_service",
    "audit_record": "security_service",
    "tombstone": "identity_service",
    "legal_hold": "legal_service",
    "marketing": "marketing_platform",
    "object": "object_storage",
    "cache": "cache_service",
    "email_delivery": "email_delivery_service",
    "telemetry": "analytics_platform",
}

# System -> default data category (``deletion_rules.csv`` vocabulary).
SYSTEM_DEFAULT_CATEGORY = {
    "account_service": "USER_PROFILE",
    "search_service": "SEARCH_INDEX_ENTRY",
    "support_platform": "SUPPORT_MESSAGE",
    "export_service": "USER_EXPORT",
    "analytics_platform": "ANALYTICS_PROFILE",
    "backup_service": "BACKUP_ARCHIVE",
    "integration_service": "DATA_FLOW",
    "security_service": "SECURITY_AUDIT_LOG",
    "identity_service": "ACCOUNT_TOMBSTONE",
    "legal_service": "LEGAL_HOLD_RECORD",
    "marketing_platform": "MARKETING_PROFILE",
    "object_storage": "SUPPORT_ATTACHMENT",
    "cache_service": "CACHE_ENTRY",
    "email_delivery_service": "EMAIL_DELIVERY_LOG",
}

# Collection names seen in snapshot exports -> data category.
COLLECTION_CATEGORY = {
    "customer_profiles": "USER_PROFILE",
    "profile_lookups": "USER_PROFILE",
    "customer_search_documents": "SEARCH_INDEX_ENTRY",
    "index_queries": "SEARCH_INDEX_ENTRY",
    "search_documents": "SEARCH_INDEX_ENTRY",
    "support_conversations": "SUPPORT_MESSAGE",
    "support_records": "SUPPORT_MESSAGE",
    "tickets": "SUPPORT_MESSAGE",
    "ticket_attachments": "SUPPORT_ATTACHMENT",
    "customer_analytics_reports": "ANALYTICS_PROFILE",
    "customer_analytics_profiles": "ANALYTICS_PROFILE",
    "profiles": "ANALYTICS_PROFILE",
    "customer_export_jobs": "USER_EXPORT",
    "customer_exports": "USER_EXPORT",
    "exports": "USER_EXPORT",
    "backup_catalog": "BACKUP_ARCHIVE",
    "backup_restore_entries": "BACKUP_ARCHIVE",
    "restore_entries": "BACKUP_ARCHIVE",
    "connector_configurations": "DATA_FLOW",
    "connector_subscriptions": "DATA_FLOW",
    "connectors": "DATA_FLOW",
    "security_audit_events": "SECURITY_AUDIT_LOG",
    "security_audit_records": "SECURITY_AUDIT_LOG",
    "audit_records": "SECURITY_AUDIT_LOG",
    "deleted_customer_references": "ACCOUNT_TOMBSTONE",
    "retained_order_records": "LEGAL_HOLD_RECORD",
}

# Record-type labels used by lookup/handover style evidence.
RECORD_TYPE_CATEGORY = {
    "USER_PROFILE": "USER_PROFILE",
    "SEARCH_INDEX_ENTRY": "SEARCH_INDEX_ENTRY",
    "SUPPORT_MESSAGE": "SUPPORT_MESSAGE",
    "SUPPORT_ATTACHMENT": "SUPPORT_ATTACHMENT",
    "ANALYTICS_PROFILE": "ANALYTICS_PROFILE",
    "PRODUCT_TELEMETRY": "PRODUCT_TELEMETRY",
    "USER_EXPORT": "USER_EXPORT",
    "MARKETING_PROFILE": "MARKETING_PROFILE",
    "EMAIL_DELIVERY_LOG": "EMAIL_DELIVERY_LOG",
    "SECURITY_AUDIT_LOG": "SECURITY_AUDIT_LOG",
    "LEGAL_HOLD_RECORD": "LEGAL_HOLD_RECORD",
    "BACKUP_ARCHIVE": "BACKUP_ARCHIVE",
    "ACCOUNT_TOMBSTONE": "ACCOUNT_TOMBSTONE",
    "DATA_FLOW": "DATA_FLOW",
    "CACHE_ENTRY": "CACHE_ENTRY",
    "SECONDARY_IDENTIFIER": "SECONDARY_IDENTIFIER",
}

PII_FIELDS = ("name", "email", "phone", "email_hash")


def system_for_resource(resource_id: str) -> Optional[str]:
    for prefix in sorted(PREFIX_SYSTEM, key=len, reverse=True):
        if resource_id.startswith(prefix + "_"):
            return PREFIX_SYSTEM[prefix]
    return None


@dataclass
class Observation:
    """One evidence item's statement about one resource at one time."""

    evidence_id: str                    # file_id from evidence_files.csv
    file_captured_at: datetime          # manifest capture time of the evidence file
    observed_at: datetime               # time the statement refers to
    system_id: str
    resource_id: str
    category: Optional[str] = None      # data category, when the evidence states or implies it
    present: Optional[bool] = None      # True = record exists, False = record absent
    pii: dict = field(default_factory=dict)        # direct identifiers present in the copy (non-null only)
    pii_fields_known: bool = False      # True when the evidence shows which identifier fields are null
    subject_ref: Optional[str] = None   # stable internal reference (user_id) kept by the record
    enabled: Optional[bool] = None      # data flows
    restore_allowed: Optional[bool] = None  # backups
    status: Optional[str] = None        # raw status label (FOUND, IDENTIFIABLE, REDACTED, ...)
    links: set = field(default_factory=set)        # identifier tokens connecting this to a subject
    source: str = ""                    # extractor + table / collection / event
    replica: Optional[str] = None       # table or replica name for database snapshots
    implicit: bool = False              # absence inferred from a scoped snapshot that omits the record
    extra: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return (self.system_id, self.resource_id)

    def has_direct_identifiers(self) -> bool:
        return any(self.pii.get(f) for f in PII_FIELDS)


@dataclass
class SnapshotCoverage:
    """A point-in-time capture that lists *every* record of a collection in scope.

    Used to prove absence: a resource of a covered system that is missing from
    the capture did not exist at ``captured_at``.
    """

    evidence_id: str
    file_captured_at: datetime
    captured_at: datetime
    system_id: str
    resource_ids: set
    source: str


@dataclass
class Activity:
    activity_id: str
    event_time: datetime
    event_type: str
    source_system: str
    target_system: str
    resource_id: str
    reported_result: str
    details: str


@dataclass
class Step:
    phase: str
    system_id: str
    resource_id: str
    operation: str
    evidence_id: str = ""

    @property
    def state(self) -> str:
        return OPERATION_STATE[self.operation]

    def sort_key(self):
        return (
            PHASES.index(self.phase),
            OPERATION_ORDER[self.phase].index(self.operation),
            self.system_id,
            self.resource_id,
        )
