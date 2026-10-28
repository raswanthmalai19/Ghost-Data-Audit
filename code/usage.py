"""Write ``code/evaluation/usage_report.md`` for the run that produced ``output.csv``.

The report states what the run actually consumed (model calls, tokens, cost per
model and overall) and, separately and clearly labelled, a projection for the
optional ``--llm`` mode computed from the same inputs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm import DEFAULT_MODEL, PRICING
from utils import CODE_DIR

USAGE_REPORT = CODE_DIR / "evaluation" / "usage_report.md"

# Projection assumptions (documented in the report).
INTENT_PROMPT_OVERHEAD = 330      # tokens of instructions around each request text
INTENT_OUTPUT = 120               # JSON label + short reason (+ minimal adaptive thinking at low effort)
VISION_PROMPT_OVERHEAD = 420
VISION_OUTPUT = 900               # JSON transcription of ~4-8 rows (+ thinking at medium effort)
MAX_IMAGE_PIXELS = 1_150_000      # images above ~1.15 MP are downscaled by the API


def _image_tokens(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as im:
        pixels = min(im.width * im.height, MAX_IMAGE_PIXELS)
    return int(pixels / 750)


def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}"


def write_usage_report(*, n_requests: int, elapsed_s: float, llm=None, low_conf_texts: list[str],
                       image_paths: list[Path], ocr_docs: int, pdf_docs: int,
                       path: Path = USAGE_REPORT, output_path: Optional[Path] = None) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = llm.usage_summary() if llm is not None else {}
    L: list[str] = []
    w = L.append
    w("# Token usage and cost - final full-dataset run")
    w("")
    w(f"_Written automatically by `python3 code/main.py` at {now} for the run that produced "
      f"`{output_path.name if output_path else 'output.csv'}`._")
    w("")
    w("## Run summary")
    w("")
    w(f"* Requests audited: **{n_requests}** (`dataset/deletion_requests.csv`)")
    w(f"* LLM layer: **{'enabled (--llm)' if llm is not None else 'disabled - deterministic mode'}**")
    w(f"* Wall-clock time: {elapsed_s:.1f} s (OCR/PDF text is cached by content hash after the first run)")
    w(f"* Local, non-LLM extraction: Tesseract OCR on {ocr_docs} screenshots, pdfplumber on {pdf_docs} PDFs, "
      "SQLite for SQL snapshots, parsers for JSON / logs / text. These cost no tokens.")
    w("")
    w("## Model usage (actual)")
    w("")
    w("| Provider | Model | Purpose | API calls | Cache hits | Input tokens | Output tokens | Total tokens | Est. cost (USD) |")
    w("|---|---|---|---|---|---|---|---|---|")
    total_calls = total_in = total_out = 0
    total_cost = 0.0
    if summary:
        for model, u in summary.items():
            tot = u["input_tokens"] + u["output_tokens"]
            w(f"| Anthropic | `{model}` | intent fallback + screenshot extraction | {u['calls']} | {u['cached_hits']} | "
              f"{u['input_tokens']:,} | {u['output_tokens']:,} | {tot:,} | {_fmt_usd(u['estimated_cost_usd'])} |")
            total_calls += u["calls"]
            total_in += u["input_tokens"]
            total_out += u["output_tokens"]
            total_cost += u["estimated_cost_usd"]
    else:
        w(f"| Anthropic | `{DEFAULT_MODEL}` (configured, not called) | intent fallback + screenshot extraction | 0 | 0 | 0 | 0 | 0 | $0.0000 |")
    w("| Local | Tesseract OCR / pdfplumber / SQLite | evidence extraction | - | - | 0 | 0 | 0 | $0.0000 |")
    w(f"| **All models** | | | **{total_calls}** | | **{total_in:,}** | **{total_out:,}** | **{total_in + total_out:,}** | **{_fmt_usd(total_cost)}** |")
    w("")
    per_req = (total_in + total_out) / n_requests if n_requests else 0
    w(f"* Average tokens per request: **{per_req:,.1f}** (input {total_in / n_requests if n_requests else 0:,.1f}, "
      f"output {total_out / n_requests if n_requests else 0:,.1f})")
    w(f"* Estimated cost per request: **{_fmt_usd(total_cost / n_requests if n_requests else 0)}**; total: **{_fmt_usd(total_cost)}**")
    w("")
    w("All decisions (identity, state at `audit_at`, policy, evidence choice, ordering) are computed deterministically, so "
      "the submitted `output.csv` does not depend on any model output; the deterministic rules already classify every "
      "request and the screenshots are read by OCR with identifier grounding.")
    w("")
    w("## Projection for the optional `--llm` mode (same inputs, not part of the submitted run)")
    w("")
    price_in, price_out = PRICING.get(DEFAULT_MODEL, (0.0, 0.0))
    intent_in = sum(INTENT_PROMPT_OVERHEAD + max(1, len(t) // 4) for t in low_conf_texts)
    intent_out = INTENT_OUTPUT * len(low_conf_texts)
    vis_in = sum(VISION_PROMPT_OVERHEAD + _image_tokens(p) for p in image_paths)
    vis_out = VISION_OUTPUT * len(image_paths)
    rows = [
        ("intent fallback (low-confidence texts only)", len(low_conf_texts), intent_in, intent_out),
        ("screenshot extraction (one call per image)", len(image_paths), vis_in, vis_out),
    ]
    w(f"Model `{DEFAULT_MODEL}` at ${price_in:.2f} / ${price_out:.2f} per million input / output tokens. Calls are made "
      "once per distinct text or image and then served from `code/cache/llm/`, so repeat runs cost nothing.")
    w("")
    w("| Purpose | Calls | Input tokens | Output tokens | Est. cost (USD) |")
    w("|---|---|---|---|---|")
    p_in = p_out = 0
    p_cost = 0.0
    for name, calls, tin, tout in rows:
        cost = tin / 1e6 * price_in + tout / 1e6 * price_out
        p_in, p_out, p_cost = p_in + tin, p_out + tout, p_cost + cost
        w(f"| {name} | {calls} | {tin:,} | {tout:,} | {_fmt_usd(cost)} |")
    w(f"| **Total** | **{sum(r[1] for r in rows)}** | **{p_in:,}** | **{p_out:,}** | **{_fmt_usd(p_cost)}** |")
    w("")
    w(f"Projected per request: {(p_in + p_out) / n_requests if n_requests else 0:,.1f} tokens, "
      f"{_fmt_usd(p_cost / n_requests if n_requests else 0)}.")
    w("")
    w(f"Assumptions: request text ~4 characters per token plus {INTENT_PROMPT_OVERHEAD} instruction tokens; image tokens "
      f"= pixels / 750 after downscaling to at most {MAX_IMAGE_PIXELS:,} pixels, plus {VISION_PROMPT_OVERHEAD} instruction "
      f"tokens; {INTENT_OUTPUT} / {VISION_OUTPUT} output tokens per intent / vision call.")
    w("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


__all__ = ["write_usage_report"]
