"""Wire the audit components together (shared by ``main.py`` and ``evaluate.py``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from activity import ActivityIndex
from audit_engine import AuditEngine, CaseResult, EngineConfig, Request, load_requests
from evidence_indexer import EvidenceIndex, build_index
from identity import IdentityResolver
from intent import IntentClassifier
from policy import PolicyEngine
from utils import DATASET_DIR


@dataclass
class Pipeline:
    dataset_dir: Path
    identities: IdentityResolver
    activity: ActivityIndex
    index: EvidenceIndex
    policy: PolicyEngine
    intent: IntentClassifier
    engine: AuditEngine
    llm: Optional[object] = None

    def run(self, requests: list[Request], context_requests: Optional[list[Request]] = None) -> list[CaseResult]:
        """Audit ``requests``.  ``context_requests`` (default: the same list) are used only for
        joint workflow attribution, so records of other requests sharing a time slot are excluded."""
        self.engine.prepare(context_requests or requests)
        return [self.engine.audit(r) for r in sorted(requests, key=lambda r: r.case_id)]


def build_pipeline(dataset_dir: Path = DATASET_DIR, use_llm: bool = False, verbose: bool = False,
                   config: Optional[EngineConfig] = None) -> Pipeline:
    llm = None
    if use_llm:
        from llm import LLMClient

        llm = LLMClient.from_env()
    identities = IdentityResolver(dataset_dir)
    activity = ActivityIndex(dataset_dir)
    index = build_index(dataset_dir, activity, identities, verbose=verbose, vlm=llm)
    policy = PolicyEngine(dataset_dir)
    intent = IntentClassifier(llm=llm)
    engine = AuditEngine(dataset_dir, index, identities, activity, policy, intent, config=config)
    return Pipeline(dataset_dir, identities, activity, index, policy, intent, engine, llm)


def all_requests(dataset_dir: Path = DATASET_DIR) -> list[Request]:
    """Every request in the participant data (evaluation + sample).  Used as attribution context."""
    reqs = []
    seen = set()
    for name in ("deletion_requests.csv", "sample_deletion_requests.csv"):
        path = dataset_dir / name
        if path.exists():
            for r in load_requests(path):
                if r.case_id not in seen:
                    seen.add(r.case_id)
                    reqs.append(r)
    return reqs


__all__ = ["Pipeline", "build_pipeline", "all_requests", "EngineConfig"]
