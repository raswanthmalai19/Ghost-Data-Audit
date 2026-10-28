"""Request-intent gate: does ``request_text`` authorise deletion?

Two layers:

1. A deterministic multilingual rule classifier (English, Spanish, French,
   Portuguese, German, Italian, Japanese, Korean, Chinese ...).  Explicit
   non-deletion signals (access / copy / export, correction, marketing opt-out,
   login help, "do not delete", "keep the account active") override deletion
   vocabulary, because those requests often *mention* deletion in order to
   rule it out ("This is an information request, not a request to erase it").
2. An optional LLM classifier (``llm.classify_intent``) used when an API key is
   configured.  It only runs on texts the rules mark as low-confidence, and its
   answers are cached by text hash so repeated runs are free and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

NON_DELETION_PATTERNS = [
    # explicit negation of deletion / closure
    r"\bnot (?:a )?request(?:ing)? (?:to|for) (?:erase|delet|remov|clos)",
    r"\bnot asking for (?:deletion|erasure|removal)",
    r"\b(?:do not|don't|dont|please don't|please do not) (?:delete|erase|remove|close|deactivate)",
    r"\bwithout (?:removing|deleting|erasing|closing)",
    r"\b(?:leave|keep) (?:the|my) (?:account|profile)(?: \w+){0,3} (?:active|open|as it is|in place)",
    r"\bkeep the account as it is\b",
    r"\bleave (?:my|the) profile\b.*\bin place\b",
    r"\b(?:i )?(?:want|would like) to (?:continue|keep) using\b",
    r"\bi still (?:use|need) (?:it|the account|this account)\b",
    r"\bstill want (?:receipts|service messages|the account)\b",
    # access / portability
    r"\bwhat personal (?:information|data) you hold\b",
    r"\b(?:send|give|provide) me a copy\b",
    r"\bcopy of (?:the|my) (?:information|data|invoices?)\b",
    r"\bdownload (?:my|a copy of my) (?:account )?(?:information|data)\b",
    r"\bportable format\b",
    r"\binformation request\b",
    # rectification
    r"\b(?:misspelled|out of date|correct (?:it|my|the)|update (?:my|the) (?:phone|number|address|name))\b",
    r"\b(?:change|replace|update) the (?:email address|mobile number|phone number)\b",
    # marketing opt-out
    r"\b(?:remove me from|unsubscribe me from) (?:the )?marketing\b",
    r"\bmarketing lists? only\b",
    r"\bstop (?:the )?(?:promotional|marketing) (?:emails|messages)\b",
    # account recovery / billing
    r"\b(?:reset(?:ting)? the login|can't access my (?:profile|account)|cannot access my (?:profile|account))\b",
    r"\binvoices? for my records\b",
    # multilingual negations
    r"\bno (?:borren|eliminen|cierren) (?:mi|la) cuenta\b",
    r"\bne (?:supprimez|fermez) pas\b",
    r"\bnão (?:excluam|apaguem|encerrem)\b",
    r"\bnicht (?:löschen|schließen)\b",
    r"削除しないで|삭제하지 마",
]

DELETION_PATTERNS = [
    r"\b(?:delet\w*|eras(?:e|ed|ing|ure)|remov(?:e|ed|al|ing)|wip(?:e|ed|ing)|purg(?:e|ed|ing))\b",
    r"\bclose (?:my|this|the) account\b",
    r"\bclosed?\b.*\baccount\b",
    r"\baccount[- ]deletion\b",
    r"\bforget me\b|\bright to be forgotten\b",
    # follow-ups on an earlier deletion request ("confirm the profile is gone")
    r"\b(?:account|profile|data) (?:is|was|has been) gone\b",
    r"\b(?:cannot|can't|can no longer|no longer) (?:identify|be linked to|point back to) me\b",
    # Spanish / Portuguese / French / Italian / German / Dutch / Polish
    r"\beliminaci[oó]n\b|\beliminar\b|\bborrar\b|\bsuprimir\b|\bcerrar mi cuenta\b",
    r"\bexclus[aã]o\b|\bexcluir\b|\bapagar\b|\bencerrar a conta\b",
    r"\bsuppression\b|\bsupprimer\b|\beffacer\b|\bfermer mon compte\b",
    r"\bcancellazione\b|\bcancellare\b|\beliminare\b",
    r"\blöschen\b|\blöschung\b|\bkonto schließen\b",
    r"\bverwijder(?:en|ing)\b|\busun(?:ąć|iecie)\b",
    # Japanese / Korean / Chinese
    r"削除|退会|消去|抹消",
    r"삭제|탈퇴",
    r"删除|注销|刪除|註銷",
]


@dataclass
class IntentResult:
    authorizes_deletion: bool
    label: str              # DELETION | NON_DELETION
    reason: str
    confidence: float
    source: str = "rules"


def classify_rules(text: str) -> IntentResult:
    t = (text or "").strip()
    low = t.lower()
    neg = [p for p in NON_DELETION_PATTERNS if re.search(p, low)]
    pos = [p for p in DELETION_PATTERNS if re.search(p, low)]
    if neg:
        return IntentResult(False, "NON_DELETION", f"non-deletion signal: {neg[0]}", 0.95 if not pos else 0.85)
    if pos:
        return IntentResult(True, "DELETION", f"deletion signal: {pos[0]}", 0.95)
    return IntentResult(False, "NON_DELETION", "no deletion authorisation found", 0.6)


class IntentClassifier:
    def __init__(self, llm=None, llm_threshold: float = 0.9):
        self.llm = llm
        self.llm_threshold = llm_threshold
        self.disagreements: list[dict] = []

    def classify(self, text: str) -> IntentResult:
        rules = classify_rules(text)
        if self.llm is None or rules.confidence >= self.llm_threshold:
            return rules
        llm_result: Optional[IntentResult] = self.llm.classify_intent(text)
        if llm_result is None:
            return rules
        if llm_result.authorizes_deletion != rules.authorizes_deletion:
            self.disagreements.append({"text": text, "rules": rules.label, "llm": llm_result.label})
        return llm_result


__all__ = ["IntentClassifier", "IntentResult", "classify_rules"]
