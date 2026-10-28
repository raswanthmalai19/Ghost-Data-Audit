# Token usage and cost - final full-dataset run

_Written automatically by `python3 code/main.py` at 2026-10-24T22:24:12Z for the run that produced `output.csv`._

## Run summary

* Requests audited: **518** (`dataset/deletion_requests.csv`)
* LLM layer: **disabled - deterministic mode**
* Wall-clock time: 0.6 s (OCR/PDF text is cached by content hash after the first run)
* Local, non-LLM extraction: Tesseract OCR on 11 screenshots, pdfplumber on 4 PDFs, SQLite for SQL snapshots, parsers for JSON / logs / text. These cost no tokens.

## Model usage (actual)

| Provider | Model | Purpose | API calls | Cache hits | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
|---|---|---|---|---|---|---|---|---|
| Anthropic | `claude-opus-5-5` (configured, not called) | intent fallback + screenshot extraction | 0 | 0 | 0 | 0 | 0 | $0.0000 |
| Local | Tesseract OCR / pdfplumber / SQLite | evidence extraction | - | - | 0 | 0 | 0 | $0.0000 |
| **All models** | | | **0** | | **0** | **0** | **0** | **$0.0000** |

* Average tokens per request: **0.0** (input 0.0, output 0.0)
* Estimated cost per request: **$0.0000**; total: **$0.0000**

All decisions (identity, state at `audit_at`, policy, evidence choice, ordering) are computed deterministically, so the submitted `output.csv` does not depend on any model output; the deterministic rules already classify every request and the screenshots are read by OCR with identifier grounding.

## Projection for the optional `--llm` mode (same inputs, not part of the submitted run)

Model `claude-opus-5-5` at $4.00 / $20.00 per million input / output tokens. Calls are made once per distinct text or image and then served from `code/cache/llm/`, so repeat runs cost nothing.

| Purpose | Calls | Input tokens | Output tokens | Est. cost (USD) |
|---|---|---|---|---|
| intent fallback (low-confidence texts only) | 6 | 2,137 | 720 | $0.0229 |
| screenshot extraction (one call per image) | 11 | 21,483 | 9,900 | $0.2839 |
| **Total** | **17** | **23,620** | **10,620** | **$0.3069** |

Projected per request: 66.1 tokens, $0.0006.

Assumptions: request text ~4 characters per token plus 330 instruction tokens; image tokens = pixels / 750 after downscaling to at most 1,150,000 pixels, plus 420 instruction tokens; 120 / 900 output tokens per intent / vision call.

