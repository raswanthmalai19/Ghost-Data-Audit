"""Evaluation harness for the deletion-audit agent.

    python3 code/evaluate.py                 # sample metrics + ablations + synthetic suite
    python3 code/evaluate.py --report        # also regenerate code/evaluation/evaluation_report.md
    python3 code/evaluate.py --diff case_23  # print a step diff for selected sample cases

Metrics (computed against ``dataset/sample_output.csv``):

* case-level ``deletion_status`` accuracy and exact ``steps_json`` match rate;
* step-level precision / recall / F1 over full step tuples
  ``(phase, system_id, resource_id, operation, state, evidence_id)``;
* field-level accuracy, both positional (step i vs step i) and on steps aligned by
  ``(phase, system_id, resource_id)``;
* sequence accuracy: share of cases whose common steps appear in the same relative
  order, plus contract validity of every emitted row.

The synthetic suite builds an independent mini-dataset (new users, new resource ids,
new evidence files including a rendered screenshot) for scenarios the public samples
do not cover, and checks the pipeline end to end.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from audit_engine import EngineConfig, load_requests  # noqa: E402
from pipeline import all_requests, build_pipeline  # noqa: E402
from steps import to_json_list, validate_steps  # noqa: E402
from utils import DATASET_DIR, email_hash  # noqa: E402

FIELDS = ("phase", "system_id", "resource_id", "operation", "state", "evidence_id")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _tuple(step: dict) -> tuple:
    return tuple(step.get(f, "") for f in FIELDS)


def score(pred: dict[str, tuple[str, list]], gold: dict[str, tuple[str, list]]) -> dict:
    n = len(gold)
    status_ok = exact = 0
    tp = fp = fn = 0
    pos_field = Counter()
    pos_total = 0
    aligned_field = Counter()
    aligned_total = 0
    seq_ok = seq_total = 0
    valid_rows = 0
    per_case = {}
    confusion = Counter()
    for cid, (gstatus, gsteps) in gold.items():
        pstatus, psteps = pred.get(cid, ("MISSING", []))
        confusion[(gstatus, pstatus)] += 1
        status_ok += pstatus == gstatus
        is_exact = pstatus == gstatus and [_tuple(s) for s in psteps] == [_tuple(s) for s in gsteps]
        exact += is_exact
        if not validate_steps(psteps, pstatus):
            valid_rows += 1
        gc, pc = Counter(map(_tuple, gsteps)), Counter(map(_tuple, psteps))
        inter = sum((gc & pc).values())
        tp += inter
        fp += sum(pc.values()) - inter
        fn += sum(gc.values()) - inter
        # positional field accuracy
        for i in range(max(len(gsteps), len(psteps))):
            pos_total += 1
            g = gsteps[i] if i < len(gsteps) else {}
            p = psteps[i] if i < len(psteps) else {}
            for f in FIELDS:
                pos_field[f] += (g.get(f) == p.get(f)) and bool(g) and bool(p)
        # aligned field accuracy (match on phase/system/resource)
        pkey = {(s["phase"], s["system_id"], s["resource_id"]): s for s in psteps}
        for g in gsteps:
            aligned_total += 1
            p = pkey.get((g["phase"], g["system_id"], g["resource_id"]))
            for f in FIELDS:
                aligned_field[f] += bool(p) and p.get(f) == g.get(f)
        # sequence: relative order of common steps
        common = [t for t in map(_tuple, gsteps) if t in set(map(_tuple, psteps))]
        porder = [t for t in map(_tuple, psteps) if t in set(common)]
        seq_total += 1
        seq_ok += common == porder
        per_case[cid] = {"status_ok": pstatus == gstatus, "exact": is_exact, "pred": pstatus, "gold": gstatus}
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": n,
        "status_accuracy": status_ok / n if n else 0,
        "exact_match": exact / n if n else 0,
        "step_precision": precision,
        "step_recall": recall,
        "step_f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "field_accuracy_positional": {f: pos_field[f] / pos_total if pos_total else 0 for f in FIELDS},
        "field_accuracy_aligned": {f: aligned_field[f] / aligned_total if aligned_total else 0 for f in FIELDS},
        "sequence_accuracy": seq_ok / seq_total if seq_total else 0,
        "contract_valid_rows": valid_rows / n if n else 0,
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
        "per_case": per_case,
    }


def load_gold(path: Path) -> dict[str, tuple[str, list]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["case_id"]: (r["deletion_status"], json.loads(r["steps_json"])) for r in csv.DictReader(fh)}


def to_pred(results) -> dict[str, tuple[str, list]]:
    return {r.case_id: (r.status, to_json_list(r.steps)) for r in results}


def evaluate_samples(config: EngineConfig | None = None, pipe=None):
    pipe = pipe or build_pipeline(config=config)
    if config is not None:
        pipe.engine.config = config
    sample = load_requests(DATASET_DIR / "sample_deletion_requests.csv")
    results = pipe.run(sample, context_requests=all_requests())
    gold = load_gold(DATASET_DIR / "sample_output.csv")
    return results, score(to_pred(results), gold), gold


def diff_case(cid: str, pred: dict, gold: dict) -> list[str]:
    gs, ps = set(map(_tuple, gold[cid][1])), set(map(_tuple, pred[cid][1]))
    out = [f"{cid}: gold={gold[cid][0]} pred={pred[cid][0]}"]
    out += [f"  missing {t}" for t in sorted(gs - ps)]
    out += [f"  extra   {t}" for t in sorted(ps - gs)]
    return out


ABLATIONS = [
    ("full system", EngineConfig()),
    ("- historical identifiers (current snapshot only)", EngineConfig(historical_identifiers=False)),
    ("- request-window scoping", EngineConfig(request_window_scope=False)),
    ("- workflow attribution", EngineConfig(workflow_attribution=False)),
    ("- snapshot absence proofs", EngineConfig(snapshot_absence=False)),
    ("- citable-evidence-only state", EngineConfig(citable_evidence_only=False)),
]


# ---------------------------------------------------------------------------
# Synthetic scenarios (independent mini-dataset)
# ---------------------------------------------------------------------------

def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        for path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _render_screenshot(path: Path, lines: list[str]) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1400, 80 + 54 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    font = _font(30)
    for i, text in enumerate(lines):
        draw.text((40, 40 + 54 * i), text, fill=(20, 24, 33), font=font)
    img.save(path)


SYNTHETIC_USERS = [
    # user_id, name, current email, phone
    ("user_901", "Aiko Fixture", "aiko.fixture@example.net", "+81 90 1111 2222"),
    ("user_902", "Bruno Fixture", "bruno.fixture@example.net", "+55 11 91111-2222"),
    ("user_903", "Dana Fixture", "dana.fixture@example.org", "+1 (212) 555-0190"),
    ("user_904", "Eira Fixture", "eira.fixture@example.org", "+46 70 111 22 33"),
    ("user_905", "Femi Fixture", "femi.fixture@example.org", "+234 803 111 2222"),
    ("user_906", "Gita Fixture", "gita.fixture@example.org", "+91 90000 11122"),
]


def build_synthetic_dataset(root: Path) -> dict[str, tuple[str, list]]:
    """Write a self-contained dataset under ``root`` and return its expected answers."""
    ev = root / "evidence"
    for sub in ("databases", "logs", "records", "images"):
        (ev / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(DATASET_DIR / "deletion_rules.csv", root / "deletion_rules.csv")

    with open(root / "user_identities.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "name", "email", "phone", "email_hash", "updated_at"])
        for uid, name, email, phone in SYNTHETIC_USERS:
            w.writerow([uid, name, email, phone, email_hash(email), "2026-03-01T00:00:00Z"])
    old_email = "aiko.f.old@example.net"
    with open(root / "user_audit_log.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"])
        w.writerow(["audit_901", "user_901", "2026-03-01T00:00:00Z", "email", old_email, "aiko.fixture@example.net"])
        w.writerow(["audit_902", "user_901", "2026-03-01T00:00:00Z", "email_hash", email_hash(old_email),
                    email_hash("aiko.fixture@example.net")])

    # S1 historical email + sync recreation loop ------------------------------
    (ev / "logs" / "synthetic_trace_20260610.log").write_text("\n".join([
        "2026-06-10T08:00:00Z level=INFO service=readback_collector event=export_started target_count=3",
        f"2026-06-10T08:01:00Z level=DEBUG service=search_service trace_id=tr-s1-1 event=index_query_started match_key={old_email}",
        "2026-06-10T08:01:02Z level=INFO service=search_service trace_id=tr-s1-1 resource_id=search_9101 event=index_query_completed result=hit",
        f"2026-06-10T08:02:00Z level=DEBUG service=account_service trace_id=tr-s1-2 event=profile_lookup_started lookup_key={old_email}",
        "2026-06-10T08:02:02Z level=INFO service=account_service trace_id=tr-s1-2 resource_id=account_9103 event=profile_lookup_completed rows=0 http_status=404",
        "2026-06-10T08:03:00Z level=DEBUG service=security_service trace_id=tr-s1-3 event=audit_record_query_started subject_ref=user_901",
        "2026-06-10T08:03:02Z level=DEBUG service=security_service trace_id=tr-s1-3 resource_id=audit_record_9104 event=audit_record_read_completed rows=1",
    ]) + "\n", encoding="utf-8")
    (ev / "records" / "synthetic_connectors_20260610.json").write_text(json.dumps({
        "exported_at": "2026-06-10T08:30:00Z", "region": "global", "source": "synthetic",
        "service_exports": {"integration_service": {"connectors": [{
            "observed_at": "2026-06-10T08:20:00Z", "resource_id": "data_flow_9102", "connector_name": "search_profile_indexer",
            "customer_email": old_email, "customer_reference": None, "enabled": True, "last_delivery_status": "delivered"}]}},
    }, indent=1), encoding="utf-8")

    # S2 regional replica drift after a false COMPLETE -------------------------
    acct = {"resource_id": "account_9201", "customer_reference": "user_902", "full_name": "Bruno Fixture",
            "email": "bruno.fixture@example.net", "phone": "+55 11 91111-2222", "account_status": "ACTIVE"}
    audit = {"resource_id": "audit_record_9202", "subject_reference": "user_902", "event_name": "account_deletion_reviewed"}
    rows = []
    for table, recs in (("customer_data_primary", [audit]), ("customer_data_replica_eu_west", [acct, audit])):
        for r in recs:
            system = "account_service" if r["resource_id"].startswith("account") else "security_service"
            coll = "customer_profiles" if system == "account_service" else "security_audit_events"
            payload = json.dumps(r).replace("'", "''")
            rows.append(f"INSERT INTO \"{table}\" VALUES('{r['resource_id']}','{system}','{coll}','{payload}','2026-06-11T09:00:00Z');")
    ddl = ("CREATE TABLE \"{t}\" (resource_id TEXT, origin_system TEXT, collection_name TEXT, record_json TEXT, "
           "updated_at TEXT, PRIMARY KEY(resource_id));")
    (ev / "databases" / "synthetic_regional_replicas_20260611.sql").write_text("\n".join(
        ["-- Captured at: 2026-06-11T09:00:00Z", "BEGIN TRANSACTION;",
         ddl.format(t="customer_data_primary"), ddl.format(t="customer_data_replica_eu_west")] + rows + ["COMMIT;"]
    ) + "\n", encoding="utf-8")

    # S3 screenshot PII exposure + legal retention hold ------------------------
    _render_screenshot(ev / "images" / "synthetic_helpline_console_20260612.png", [
        "Helpline Console   /   Inbox",
        "CS-9301  Where is my parcel?",
        "support_9301   Dana Fixture   Normal priority",
        "Email   dana.fixture@example.org",
        "09:12  Dana Fixture: please call me before delivery",
        "09:20  Agent: the parcel reached the regional hub",
    ])
    (ev / "records" / "synthetic_bundle_20260612.json").write_text(json.dumps({
        "exported_at": "2026-06-12T10:00:00Z", "source": "synthetic",
        "snapshots": [
            {"captured_at": "2026-06-12T10:00:00Z", "data_store": "account_service", "collection": "customer_profiles",
             "records": []},
            {"captured_at": "2026-06-12T10:00:00Z", "data_store": "legal_service", "collection": "retained_order_records",
             "records": [{"resource_id": "legal_hold_9302", "subject_reference": "user_903", "order_reference": "ORD-9302",
                          "retention_reason": "payment_dispute_and_tax_record", "retained_until": "2033-12-31T23:59:59Z"}]},
        ]}, indent=1), encoding="utf-8")

    # S4 temporal trap: a snapshot captured after audit_at must not change the answer
    (ev / "records" / "synthetic_export_20260613.json").write_text(json.dumps({
        "exported_at": "2026-06-13T11:00:00Z", "region": "global", "source": "synthetic",
        "service_exports": {
            "account_service": {"profile_lookups": [
                {"observed_at": "2026-06-13T10:00:00Z", "profile": None, "resource_id": "account_9401", "status_code": 404}]},
            "analytics_platform": {"profiles": [
                {"observed_at": "2026-06-13T10:05:00Z", "resource_id": "analytics_9403", "customer_reference": None,
                 "email": None, "phone": None, "email_hash": None, "audience_cohort": "general_shopper"}]},
            "security_service": {"audit_records": [
                {"observed_at": "2026-06-13T10:10:00Z", "resource_id": "audit_record_9402", "subject_reference": "user_904",
                 "event_code": "ACCOUNT_DELETION_RECORDED", "retention_until": "2031-12-31T23:59:59Z"}]},
        }}, indent=1), encoding="utf-8")
    (ev / "records" / "synthetic_late_bundle_20260620.json").write_text(json.dumps({
        "exported_at": "2026-06-20T12:00:00Z", "source": "synthetic",
        "snapshots": [{"captured_at": "2026-06-20T12:00:00Z", "data_store": "account_service", "collection": "customer_profiles",
                       "records": [{"resource_id": "account_9401", "customer_reference": "user_904", "full_name": "Eira Fixture",
                                    "email": "eira.fixture@example.org", "phone": "+46 70 111 22 33", "account_status": "ACTIVE"}]}],
    }, indent=1), encoding="utf-8")

    # S6 tombstone with residual PII + restorable backup ------------------------
    (ev / "records" / "synthetic_tombstones_20260614.json").write_text(json.dumps({
        "exported_at": "2026-06-14T12:00:00Z", "source": "synthetic",
        "snapshots": [
            {"captured_at": "2026-06-14T12:00:00Z", "data_store": "identity_service", "collection": "deleted_customer_references",
             "records": [{"resource_id": "tombstone_9601", "subject_reference": "user_906", "name": None,
                          "email": "gita.fixture@example.org", "phone": None, "email_hash": None, "account_status": "DELETED"}]},
            {"captured_at": "2026-06-14T12:00:00Z", "data_store": "backup_service", "collection": "backup_catalog",
             "records": [{"resource_id": "backup_9602", "archive_path": "s3://b/segment-96.tar.zst", "subject_reference": "user_906",
                          "restore_allowed": True, "encryption": "AES-256"}]},
            {"captured_at": "2026-06-14T12:00:00Z", "data_store": "account_service", "collection": "customer_profiles",
             "records": []},
        ]}, indent=1), encoding="utf-8")

    manifest = [
        ("evidence_s1", "evidence/logs/synthetic_trace_20260610.log", "2026-06-10T08:10:00Z"),
        ("evidence_s2", "evidence/records/synthetic_connectors_20260610.json", "2026-06-10T08:30:00Z"),
        ("evidence_s3", "evidence/databases/synthetic_regional_replicas_20260611.sql", "2026-06-11T09:00:00Z"),
        ("evidence_s4", "evidence/images/synthetic_helpline_console_20260612.png", "2026-06-12T09:30:00Z"),
        ("evidence_s5", "evidence/records/synthetic_bundle_20260612.json", "2026-06-12T10:00:00Z"),
        ("evidence_s6", "evidence/records/synthetic_export_20260613.json", "2026-06-13T11:00:00Z"),
        ("evidence_s7", "evidence/records/synthetic_late_bundle_20260620.json", "2026-06-20T12:00:00Z"),
        ("evidence_s8", "evidence/records/synthetic_tombstones_20260614.json", "2026-06-14T12:00:00Z"),
    ]
    with open(root / "evidence_files.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file_id", "file_path", "origin", "captured_at"])
        for fid, path, ts in manifest:
            w.writerow([fid, path, "synthetic_fixture", ts])

    acts = [
        # S1: primary profile submitted 60 min before reported_at; sync job recreated the search entry
        ("activity_s1", "2026-06-09T11:00:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_9103", "SUCCESS",
         "Deletion worker submitted primary profile account_9103 to account_service."),
        ("activity_s2", "2026-06-09T13:10:00Z", "SYNC_RECORD", "integration_service", "search_service", "data_flow_9102", "SUCCESS",
         "Active connector repopulated downstream resource search_9101."),
        # S3: account deleted, attributed only through the workflow window
        ("activity_s3", "2026-06-11T10:55:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_9303", "SUCCESS",
         "Deletion worker submitted primary profile account_9303 to account_service."),
        # S4: the deleted account (now a bare 404) and an identifier-less analytics profile are
        # connected to the request only through its own workflow events
        ("activity_s6", "2026-06-12T11:05:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_9401", "SUCCESS",
         "Deletion worker submitted primary profile account_9401 to account_service."),
        ("activity_s4", "2026-06-12T11:31:00Z", "ANONYMIZE_RECORD", "deletion_worker", "analytics_platform", "analytics_9403", "SUCCESS",
         "Policy action completed for analytics_9403."),
        # S6
        ("activity_s5", "2026-06-13T10:50:00Z", "DELETE_RECORD", "deletion_worker", "account_service", "account_9603", "PARTIAL",
         "Deletion worker submitted primary profile account_9603 to account_service."),
    ]
    with open(root / "system_activity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["activity_id", "event_time", "event_type", "source_system", "target_system", "resource_id",
                    "reported_result", "details"])
        w.writerows(acts)

    requests = [
        ("case_s1", "aiko.fixture@example.net", "2026-06-08T12:00:00Z", "COMPLETE", "2026-06-09T12:00:00Z",
         "2026-06-10T12:00:00Z", "Please delete my account. I changed my email earlier, so check the old address as well."),
        ("case_s2", "bruno.fixture@example.net", "2026-06-09T12:00:00Z", "COMPLETE", "2026-06-10T12:00:00Z",
         "2026-06-11T12:00:00Z", "Please permanently delete the account associated with this email address."),
        ("case_s3", "dana.fixture@example.org", "2026-06-10T12:00:00Z", "COMPLETE", "2026-06-11T12:00:00Z",
         "2026-06-12T12:00:00Z", "Erase my account and personal data. Keep only what the law requires."),
        ("case_s4", "eira.fixture@example.org", "2026-06-11T12:00:00Z", "COMPLETE", "2026-06-12T12:00:00Z",
         "2026-06-13T12:00:00Z", "Je demande la suppression définitive du compte associé à cette adresse e-mail."),
        ("case_s5", "femi.fixture@example.org", "2026-06-12T12:00:00Z", "NOT_STARTED", "2026-06-13T12:00:00Z",
         "2026-06-14T12:00:00Z", "Por favor, no borren mi cuenta; solo quiero una copia de mis datos."),
        ("case_s6", "gita.fixture@example.org", "2026-06-12T12:00:00Z", "PARTIALLY_COMPLETE", "2026-06-13T12:00:00Z",
         "2026-06-14T12:00:00Z", "Please close my account and remove my personal details from anything you keep."),
    ]
    for name in ("sample_deletion_requests.csv", "deletion_requests.csv"):
        with open(root / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "email", "requested_at", "reported_status", "reported_at", "audit_at", "request_text"])
            w.writerows(requests)

    def s(phase, system, rid, op, ev=""):
        return {"phase": phase, "system_id": system, "resource_id": rid, "operation": op, "evidence_id": ev}

    expected = {
        "case_s1": ("INCOMPLETE", [
            s("FINDING", "search_service", "search_9101", "REPORT_REMAINING_RECORD", "evidence_s1"),
            s("FINDING", "integration_service", "data_flow_9102", "REPORT_ACTIVE_DATA_FLOW", "evidence_s2"),
            s("REMEDIATION", "integration_service", "data_flow_9102", "DISABLE_DATA_FLOW"),
            s("REMEDIATION", "search_service", "search_9101", "DELETE_RECORD"),
            s("VERIFICATION", "account_service", "account_9103", "VERIFY_RECORD_ABSENT"),
            s("VERIFICATION", "search_service", "search_9101", "VERIFY_RECORD_ABSENT"),
            s("VERIFICATION", "security_service", "audit_record_9104", "VERIFY_RECORD_PRESENT"),
            s("VERIFICATION", "integration_service", "data_flow_9102", "VERIFY_DATA_FLOW_DISABLED"),
        ]),
        "case_s2": ("INCOMPLETE", [
            s("FINDING", "account_service", "account_9201", "REPORT_REMAINING_RECORD", "evidence_s3"),
            s("REMEDIATION", "account_service", "account_9201", "DELETE_RECORD"),
            s("VERIFICATION", "account_service", "account_9201", "VERIFY_RECORD_ABSENT"),
            s("VERIFICATION", "security_service", "audit_record_9202", "VERIFY_RECORD_PRESENT"),
        ]),
        "case_s3": ("INCOMPLETE", [
            s("FINDING", "support_platform", "support_9301", "REPORT_EXPOSED_PII", "evidence_s4"),
            s("REMEDIATION", "support_platform", "support_9301", "REDACT_RECORD"),
            s("VERIFICATION", "account_service", "account_9303", "VERIFY_RECORD_ABSENT"),
            s("VERIFICATION", "legal_service", "legal_hold_9302", "VERIFY_RECORD_PRESENT"),
            s("VERIFICATION", "support_platform", "support_9301", "VERIFY_RECORD_REDACTED"),
        ]),
        "case_s4": ("COMPLETE", [
            s("VERIFICATION", "account_service", "account_9401", "VERIFY_RECORD_ABSENT", "evidence_s6"),
            s("VERIFICATION", "security_service", "audit_record_9402", "VERIFY_RECORD_PRESENT", "evidence_s6"),
            s("VERIFICATION", "analytics_platform", "analytics_9403", "VERIFY_RECORD_ANONYMIZED", "evidence_s6"),
        ]),
        "case_s5": ("NOT_APPLICABLE", []),
        "case_s6": ("INCOMPLETE", [
            s("FINDING", "identity_service", "tombstone_9601", "REPORT_EXPOSED_PII", "evidence_s8"),
            s("FINDING", "backup_service", "backup_9602", "REPORT_RESTORE_RISK", "evidence_s8"),
            s("REMEDIATION", "backup_service", "backup_9602", "BLOCK_RESTORE"),
            s("REMEDIATION", "identity_service", "tombstone_9601", "REDACT_RECORD"),
            s("VERIFICATION", "account_service", "account_9603", "VERIFY_RECORD_ABSENT"),
            s("VERIFICATION", "identity_service", "tombstone_9601", "VERIFY_RECORD_REDACTED"),
            s("VERIFICATION", "backup_service", "backup_9602", "VERIFY_RESTORE_BLOCKED"),
        ]),
    }
    from model import OPERATION_STATE

    out = {}
    for cid, (status, steps) in expected.items():
        full = []
        for i, st in enumerate(steps, start=1):
            full.append({"step_no": i, **st, "state": OPERATION_STATE[st["operation"]]})
        out[cid] = (status, full)
    return out


SYNTHETIC_DESCRIPTIONS = {
    "case_s1": "Historical email + sync recreation loop: the request uses the new address; the search entry and the "
               "connector that keeps recreating it are only reachable through the old address (log trace + connector export).",
    "case_s2": "Regional replica drift after a false COMPLETE claim: the primary table no longer has the account but the "
               "eu-west replica still does.",
    "case_s3": "Screenshot PII exposure + legal retention hold: a rendered support-console screenshot (OCR) exposes the "
               "subject's email; a legal-hold record must be retained; the deleted account is attributed via the workflow.",
    "case_s4": "Temporal trap + identifier-less anonymised profile: a snapshot captured after audit_at shows the account "
               "again and must be ignored; the anonymised analytics record is attributed through the request's workflow.",
    "case_s5": "Non-deletion request in Spanish (\"no borren mi cuenta; solo quiero una copia\") must be NOT_APPLICABLE.",
    "case_s6": "Account tombstone with a residual email + restorable backup: redact the tombstone, block the restore, "
               "block before changing dependent records.",
}


def run_synthetic(keep_dir: Path | None = None):
    tmp = Path(tempfile.mkdtemp(prefix="ghost_synth_"))
    try:
        expected = build_synthetic_dataset(tmp)
        pipe = build_pipeline(tmp)
        reqs = load_requests(tmp / "deletion_requests.csv")
        results = pipe.run(reqs, context_requests=reqs)
        pred = to_pred(results)
        metrics = score(pred, expected)
        diffs = [line for cid in expected if not metrics["per_case"][cid]["exact"] for line in diff_case(cid, pred, expected)]
        if keep_dir:
            shutil.copytree(tmp, keep_dir, dirs_exist_ok=True)
        return metrics, diffs, pred
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="regenerate code/evaluation/evaluation_report.md")
    ap.add_argument("--diff", nargs="*", default=None, help="sample case ids to diff (default: all mismatches)")
    ap.add_argument("--no-ablations", action="store_true")
    args = ap.parse_args(argv)

    pipe = build_pipeline()
    results, metrics, gold = evaluate_samples(pipe=pipe)
    pred = to_pred(results)
    print("== Sample evaluation (dataset/sample_output.csv) ==")
    for k in ("cases", "status_accuracy", "exact_match", "step_precision", "step_recall", "step_f1",
              "sequence_accuracy", "contract_valid_rows"):
        v = metrics[k]
        print(f"  {k:22s} {v if isinstance(v, int) else _pct(v)}")
    print("  field accuracy (aligned):", {k: _pct(v) for k, v in metrics["field_accuracy_aligned"].items()})
    print("  field accuracy (positional):", {k: _pct(v) for k, v in metrics["field_accuracy_positional"].items()})
    targets = args.diff if args.diff else [c for c, v in metrics["per_case"].items() if not v["exact"]]
    for cid in targets:
        for line in diff_case(cid, pred, gold):
            print("  " + line)

    ablation_rows = []
    if not args.no_ablations:
        print("\n== Ablations (sample set) ==")
        for name, cfg in ABLATIONS:
            pipe.engine.config = cfg
            _, m, _ = evaluate_samples(pipe=pipe)
            ablation_rows.append((name, m))
            print(f"  {name:50s} status={_pct(m['status_accuracy'])} exact={_pct(m['exact_match'])} "
                  f"F1={_pct(m['step_f1'])}")
        pipe.engine.config = EngineConfig()

    print("\n== Synthetic scenarios ==")
    syn_metrics, syn_diffs, syn_pred = run_synthetic()
    for cid, v in syn_metrics["per_case"].items():
        print(f"  {cid}: {'PASS' if v['exact'] else 'FAIL'}  ({SYNTHETIC_DESCRIPTIONS[cid][:70]}...)")
    for line in syn_diffs:
        print("  " + line)
    print(f"  synthetic exact match: {_pct(syn_metrics['exact_match'])}  step F1: {_pct(syn_metrics['step_f1'])}")

    if args.report:
        from report import write_evaluation_report

        write_evaluation_report(metrics, gold, pred, ablation_rows, syn_metrics, syn_pred)
        print("\nwrote code/evaluation/evaluation_report.md")
    ok = syn_metrics["exact_match"] == 1.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
