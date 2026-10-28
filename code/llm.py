"""Optional Claude layer: intent classification and screenshot extraction.

The audit is fully deterministic without this module.  When enabled
(``python3 code/main.py --llm`` with Anthropic credentials in the environment),
Claude is used in exactly two *describe-only* places:

1. ``classify_intent`` - only for request texts the multilingual rules mark as
   low-confidence;
2. ``vision_extract`` - reads application screenshots into a JSON description
   (identifiers and visible state).  Its identifiers are cross-checked against
   OCR text and participant data before anything reaches the engine.

Every response is cached on disk (``code/cache/llm/``) keyed by model, prompt
version and input hash, so repeated runs are free and reproducible.  Token
usage per model is recorded for ``evaluation/usage_report.md``.  Prompts live in
``code/prompts/``.  No credential is ever read from or written to the repo.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils import CACHE_DIR, CODE_DIR

DEFAULT_MODEL = os.environ.get("GHOST_LLM_MODEL", "claude-opus-5-5")
PROMPT_VERSION = "v1"

# USD per million tokens (Anthropic first-party API list prices).
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["DELETION", "NON_DELETION"]},
        "request_type": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["label", "request_type", "reason"],
    "additionalProperties": False,
}

VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "screen_type": {"type": "string"},
        "records": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "emails": {"type": "array", "items": {"type": "string"}},
                    "user_ids": {"type": "array", "items": {"type": "string"}},
                    "status_text": {"type": "string"},
                    "is_selected": {"type": "boolean"},
                },
                "required": ["resource_id", "emails", "user_ids", "status_text", "is_selected"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["screen_type", "records"],
    "additionalProperties": False,
}


def _prompt(name: str) -> str:
    return (CODE_DIR / "prompts" / name).read_text(encoding="utf-8")


@dataclass
class Usage:
    calls: int = 0
    cached_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    refusals: int = 0
    errors: int = 0


@dataclass
class LLMClient:
    client: object
    model: str = DEFAULT_MODEL
    cache_dir: Path = CACHE_DIR / "llm"
    usage: dict = field(default_factory=lambda: defaultdict(Usage))

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "LLMClient":
        import anthropic  # imported lazily: the deterministic pipeline never needs it

        return cls(client=anthropic.Anthropic(), model=model or DEFAULT_MODEL)

    # ------------------------------------------------------------------ core
    def _cache_path(self, kind: str, payload: str) -> Path:
        key = hashlib.sha256(f"{self.model}|{PROMPT_VERSION}|{kind}|{payload}".encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / kind / f"{key}.json"

    def _call(self, kind: str, cache_payload: str, content: list, schema: dict, effort: str,
              max_tokens: int) -> Optional[dict]:
        path = self._cache_path(kind, cache_payload)
        if path.exists():
            self.usage[self.model].cached_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["result"]
        stats = self.usage[self.model]
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                betas=["server-side-fallback-2026-07-01"],
                messages=[{"role": "user", "content": content}],
                # passed through extra_body so the call also works on SDK builds that
                # predate the named output_config / fallbacks parameters
                extra_body={
                    "fallbacks": "default",
                    "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
                },
            )
        except Exception as exc:  # network / auth / rate limit: degrade to deterministic path
            stats.errors += 1
            print(f"[llm] {kind} call failed: {type(exc).__name__}: {exc}")
            return None
        stats.calls += 1
        u = getattr(response, "usage", None)
        if u is not None:
            stats.input_tokens += getattr(u, "input_tokens", 0) or 0
            stats.output_tokens += getattr(u, "output_tokens", 0) or 0
            stats.cache_read_input_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
            stats.cache_creation_input_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
        if getattr(response, "stop_reason", None) == "refusal":
            stats.refusals += 1
            return None
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), None)
        if not text:
            return None
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            stats.errors += 1
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "prompt_version": PROMPT_VERSION, "result": result},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
        return result

    # -------------------------------------------------------------- intent
    def classify_intent(self, text: str):
        from intent import IntentResult

        prompt = _prompt("intent_classifier.md").replace("{{REQUEST_TEXT}}", text)
        result = self._call("intent", text, [{"type": "text", "text": prompt}], INTENT_SCHEMA,
                            effort="low", max_tokens=2048)
        if not result or result.get("label") not in {"DELETION", "NON_DELETION"}:
            return None
        return IntentResult(result["label"] == "DELETION", result["label"],
                            f"llm:{result.get('request_type')}: {result.get('reason', '')[:120]}", 0.9, source="llm")

    # -------------------------------------------------------------- vision
    def vision_extract(self, image_path: Path) -> Optional[dict]:
        data = image_path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        media = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": media,
                                         "data": base64.standard_b64encode(data).decode("utf-8")}},
            {"type": "text", "text": _prompt("screenshot_extraction.md")},
        ]
        return self._call("vision", digest, content, VISION_SCHEMA, effort="medium", max_tokens=8000)

    # -------------------------------------------------------------- usage
    def usage_summary(self) -> dict:
        out = {}
        for model, u in self.usage.items():
            price_in, price_out = PRICING.get(model, (0.0, 0.0))
            cost = u.input_tokens / 1e6 * price_in + u.output_tokens / 1e6 * price_out
            out[model] = {**u.__dict__, "estimated_cost_usd": round(cost, 6),
                          "price_per_mtok_in": price_in, "price_per_mtok_out": price_out}
        return out

    def write_usage(self, path: Path, n_requests: int) -> None:
        summary = self.usage_summary()
        total_in = sum(v["input_tokens"] for v in summary.values())
        total_out = sum(v["output_tokens"] for v in summary.values())
        total_cost = sum(v["estimated_cost_usd"] for v in summary.values())
        payload = {
            "requests": n_requests,
            "per_model": summary,
            "total": {"calls": sum(v["calls"] for v in summary.values()), "input_tokens": total_in,
                      "output_tokens": total_out, "estimated_cost_usd": round(total_cost, 6)},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


__all__ = ["LLMClient", "PRICING", "DEFAULT_MODEL"]
