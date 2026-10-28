"""System activity stream: claims, data-catalog links and deletion-workflow attribution.

``system_activity.csv`` is a global stream that mixes every subject's events.
Its ``reported_result`` values are *claims*; they never prove a state.  The
stream is still useful in three ways:

1. **Data-catalog links** (``LINK_RESOURCE_IDENTITY``) tie screenshots and PDFs
   (by alias: file name, statement number, customer key) to a resource, an
   email and a user id.
2. **Treatment intent**: the operation the deletion worker attempted on a
   resource (DELETE / REDACT / ANONYMIZE / BLOCK / DISABLE) reveals the policy
   treatment the workflow applied to that record type.
3. **Workflow attribution**: every deletion request's workflow submits the
   subject's primary profile and applies policy actions in a fixed window
   around the request's ``reported_at``.  Records that no longer carry any
   identifier (a deleted account that now returns 404, an analytics profile
   that was anonymised) can only be connected to the subject through this
   window.  ``WorkflowAttributor`` solves that assignment jointly for all
   requests that share a window, excluding records positively linked to other
   subjects.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from model import Activity
from utils import norm_email, norm_user, parse_ts

CATALOG_RE = re.compile(
    r"(?:Data|Attachment) catalog (?:linked|confirmed) (?P<rid>\S+) \((?P<alias>[^)]*)\) "
    r"(?:to|belongs only to) (?P<email>\S+@\S+?) and (?P<user>user_\d+)"
)

ACTION_TREATMENT = {
    "DELETE_RECORD": "DELETE",
    "REDACT_RECORD": "REDACT",
    "ANONYMIZE_RECORD": "ANONYMIZE",
    "BLOCK_RESTORE": "BLOCK_RESTORE",
    "DISABLE_DATA_FLOW": "DISABLE_DATA_FLOW",
    "RETAIN_RECORD": "RETAIN",
    "CHECK_DATA_FLOW": "DISABLE_DATA_FLOW",
    "CHECK_RESTORE_ACCESS": "BLOCK_RESTORE",
}

# reported_status of a request -> result the workflow logged for its primary profile
STATUS_RESULT = {"COMPLETE": "SUCCESS", "PARTIALLY_COMPLETE": "PARTIAL", "NOT_STARTED": "FAILED"}


@dataclass
class CatalogLink:
    activity_id: str
    event_time: datetime
    system_id: str
    resource_id: str
    alias: str
    email: Optional[str]
    user_id: Optional[str]


class ActivityIndex:
    def __init__(self, dataset_dir: Path):
        self.activities: list[Activity] = []
        with open(dataset_dir / "system_activity.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                self.activities.append(
                    Activity(
                        activity_id=row["activity_id"],
                        event_time=parse_ts(row["event_time"]),
                        event_type=row["event_type"],
                        source_system=row["source_system"],
                        target_system=row["target_system"],
                        resource_id=row["resource_id"],
                        reported_result=row["reported_result"],
                        details=row["details"],
                    )
                )
        self.activities.sort(key=lambda a: (a.event_time, a.activity_id))
        self.by_resource: dict[str, list[Activity]] = defaultdict(list)
        for a in self.activities:
            self.by_resource[a.resource_id].append(a)
            m = re.search(r"downstream resource (\S+?)\.?$", a.details)
            if m:
                self.by_resource[m.group(1)].append(a)
        self.catalog: list[CatalogLink] = []
        for a in self.activities:
            if a.event_type != "LINK_RESOURCE_IDENTITY":
                continue
            m = CATALOG_RE.search(a.details)
            if not m:
                continue
            self.catalog.append(
                CatalogLink(
                    activity_id=a.activity_id,
                    event_time=a.event_time,
                    system_id=a.target_system,
                    resource_id=m.group("rid"),
                    alias=m.group("alias"),
                    email=norm_email(m.group("email").rstrip(".;")),
                    user_id=norm_user(m.group("user")),
                )
            )

    @property
    def resource_ids(self) -> set[str]:
        out = set()
        for a in self.activities:
            if re.fullmatch(r"[a-z_]+_\d+", a.resource_id):
                out.add(a.resource_id)
            m = re.search(r"downstream resource (\S+?)\.?$", a.details)
            if m:
                out.add(m.group(1))
        return out

    def system_of(self, resource_id: str) -> Optional[str]:
        for a in self.by_resource.get(resource_id, []):
            if a.resource_id == resource_id and a.target_system not in {
                "account_replication_cluster", "account_failover_replica", "privacy_case_management", "archive_service",
                "object_store",
            }:
                return a.target_system
        return None

    def treatment_intent(self, resource_id: str, before: Optional[datetime] = None) -> Optional[str]:
        """The policy treatment the deletion worker attempted on a resource (a claim, not proof)."""
        intent = None
        for a in self.by_resource.get(resource_id, []):
            if a.resource_id != resource_id or a.source_system != "deletion_worker":
                continue
            if before is not None and a.event_time > before:
                continue
            intent = ACTION_TREATMENT.get(a.event_type, intent)
        return intent

    def non_deletion_classifications(self) -> dict[str, Activity]:
        return {a.resource_id: a for a in self.activities if a.event_type == "CLASSIFY_REQUEST"}

    def recreation_events(self, resource_id: str, until: datetime) -> list[Activity]:
        """Sync / restore jobs that repopulated a resource (evidence of an active recreation path)."""
        out = []
        for a in self.by_resource.get(resource_id, []):
            if a.event_time <= until and ("repopulated" in a.details or a.event_type == "RESTORE_BACKUP"):
                out.append(a)
        return out


# ---------------------------------------------------------------------------
# Workflow attribution
# ---------------------------------------------------------------------------

PRIMARY_PROFILE_RE = re.compile(r"Deletion worker submitted primary profile (\S+) to account_service")
TARGET_RESOURCE_RE = re.compile(r"Target resource: (\S+?)\.?$")

# Offsets (minutes, relative to reported_at) observed for the workflow events.
PRIMARY_WINDOW = (-80.0, -40.0)
COMPLETED_ACTION_WINDOW = (-36.0, 0.0)
FOLLOW_UP_WINDOW = (25.0, 125.0)
NARRATIVE_WINDOW = (-12.0, 12.0)


@dataclass
class WorkflowEvent:
    activity: Activity
    kind: str            # primary_profile | completed_action | follow_up | narrative
    offset_min: float


class WorkflowAttributor:
    """Assign identifier-less workflow events to the request whose window they fall in."""

    def __init__(self, index: ActivityIndex):
        self.index = index
        self.events: list[WorkflowEvent] = []

    def candidates(self, reported_at: datetime) -> list[WorkflowEvent]:
        lo = reported_at + timedelta(minutes=PRIMARY_WINDOW[0] - 1)
        hi = reported_at + timedelta(minutes=FOLLOW_UP_WINDOW[1] + 1)
        out = []
        for a in self.index.activities:
            if a.event_time < lo or a.event_time > hi or a.source_system != "deletion_worker":
                continue
            off = (a.event_time - reported_at).total_seconds() / 60.0
            if PRIMARY_PROFILE_RE.search(a.details) and PRIMARY_WINDOW[0] <= off <= PRIMARY_WINDOW[1]:
                out.append(WorkflowEvent(a, "primary_profile", off))
            elif TARGET_RESOURCE_RE.search(a.details) and NARRATIVE_WINDOW[0] <= off <= NARRATIVE_WINDOW[1]:
                out.append(WorkflowEvent(a, "narrative", off))
            elif a.details.startswith("Policy action completed") and COMPLETED_ACTION_WINDOW[0] <= off <= COMPLETED_ACTION_WINDOW[1]:
                out.append(WorkflowEvent(a, "completed_action", off))
            elif FOLLOW_UP_WINDOW[0] <= off <= FOLLOW_UP_WINDOW[1] and (
                "final result" in a.details or a.event_type in {"CHECK_DATA_FLOW", "CHECK_RESTORE_ACCESS"}
            ):
                out.append(WorkflowEvent(a, "follow_up", off))
        return out


__all__ = ["ActivityIndex", "CatalogLink", "WorkflowAttributor", "STATUS_RESULT", "ACTION_TREATMENT"]
