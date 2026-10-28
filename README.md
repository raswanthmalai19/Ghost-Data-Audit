# Ghost Data Audit

Ghost Data Audit is a deterministic Python application for verifying whether
customer data has actually been removed after a deletion request. It resolves
current and historical identifiers, correlates evidence across primary systems,
replicas, exports, logs, attachments, search indexes, analytics, backups, and
data flows, then produces an evidence-backed remediation plan.

The application is designed for repeatable results: evidence extraction is
separated from decision-making, policy evaluation is table-driven, and output
steps are normalized into a stable order.

## What it does

For every request, the pipeline:

1. Determines whether the message authorizes deletion.
2. Resolves the subject's current and historical identifiers.
3. Indexes participant-facing evidence from structured files, SQL snapshots,
   logs, documents, and images.
4. Reconstructs resource state at the request's audit time.
5. Applies the supplied deletion policy, including retention, redaction,
   anonymization, tombstone, restore, and synchronization rules.
6. Writes findings, ordered remediation actions, and verification checks.

The result is written to `output.csv` with the exact columns:

```text
case_id,deletion_status,steps_json
```

## Requirements

- Python 3.10 or newer
- Tesseract OCR for screenshot evidence

Install the Python dependencies:

```bash
python3 -m pip install -r code/requirements.txt
```

Install Tesseract if it is not already available:

```bash
# macOS
brew install tesseract

# Debian or Ubuntu
sudo apt-get install tesseract-ocr
```

## Run

From the repository root:

```bash
python3 code/main.py
```

The command reads `dataset/deletion_requests.csv` and writes the completed
`output.csv`. A quiet run is available for automation:

```bash
python3 code/main.py --quiet
```

To run against the public sample:

```bash
python3 code/main.py \
  --input dataset/sample_deletion_requests.csv \
  --output /tmp/sample_output.csv \
  --quiet
```

Optional commands:

```bash
# Compare the implementation with the public sample and refresh the report
python3 code/evaluate.py --report

# Inspect differences for one sample case
python3 code/evaluate.py --diff case_23

# Write per-case diagnostic traces
python3 code/main.py --trace traces.jsonl
```

An optional language-model and vision layer can be enabled with `--llm` when
the required provider credentials are supplied through environment variables.
The default run does not require a model call.

## Project layout

```text
code/
  main.py                 command-line entry point
  pipeline.py             component wiring and execution
  identity.py             current and historical identity resolution
  evidence_indexer.py     evidence loading and indexing
  extractors/             SQL, JSON, log, document, and image readers
  activity.py             activity and catalog correlation
  attribution.py          identifier-free workflow attribution
  audit_engine.py         state reconstruction and audit decisions
  policy.py               deletion-policy evaluation
  steps.py                canonical step ordering and validation
  evaluate.py             sample evaluation and diagnostics
  evaluation/             evaluation and usage reports
  prompts/                optional model prompts
  requirements.txt        Python dependencies

dataset/
  deletion_requests.csv   evaluation requests
  evidence/               participant-facing evidence
  *.csv                   identities, policy, activity, and manifests
```

## Design principles

- **Evidence before claims:** workflow statuses and activity results are
  treated as claims unless participant-facing evidence proves the state.
- **Historical coverage:** old email addresses, hashes, phone numbers, and
  stable user references remain searchable when they are relevant to the audit.
- **Temporal correctness:** evidence captured after `audit_at` cannot affect a
  case.
- **Policy fidelity:** retained records remain available, while prohibited
  identifiers are redacted or anonymized as required.
- **Deterministic output:** all steps use the allowed operations, stable
  identifiers, consecutive numbering, and canonical ordering.
