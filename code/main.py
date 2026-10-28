"""Ghost Data Still Exists - deletion-audit agent (entry point).

Usage (from the repository root):

    python3 code/main.py                       # audit dataset/deletion_requests.csv -> ./output.csv
    python3 code/main.py --input dataset/sample_deletion_requests.csv --output /tmp/sample_out.csv
    python3 code/main.py --llm                 # also use the optional Claude intent/vision layer
    python3 code/main.py --trace traces.jsonl  # write per-case reasoning traces

Pipeline: identity resolution -> once-per-run evidence index (SQL / JSON / logs /
text / PDF / OCR) -> joint workflow attribution -> per-case state reconstruction
at ``audit_at`` -> policy evaluation (``deletion_rules.csv``) -> canonical steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from audit_engine import load_requests  # noqa: E402
from pipeline import all_requests, build_pipeline  # noqa: E402
from steps import serialize, to_json_list, validate_steps  # noqa: E402
from utils import DATASET_DIR, REPO_ROOT  # noqa: E402


def write_output(results, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["case_id", "deletion_status", "steps_json"])
        for res in sorted(results, key=lambda r: _case_sort_key(r.case_id)):
            writer.writerow([res.case_id, res.status, serialize(res.steps)])


def _case_sort_key(case_id: str):
    head, _, num = case_id.rpartition("_")
    return (head, int(num)) if num.isdigit() else (case_id, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit deletion requests and write output.csv")
    ap.add_argument("--input", default=str(DATASET_DIR / "deletion_requests.csv"))
    ap.add_argument("--output", default=str(REPO_ROOT / "output.csv"))
    ap.add_argument("--dataset", default=str(DATASET_DIR))
    ap.add_argument("--llm", action="store_true", help="enable the optional Claude intent/vision layer")
    ap.add_argument("--trace", default=None, help="write per-case JSONL traces to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.time()
    dataset_dir = Path(args.dataset)
    pipe = build_pipeline(dataset_dir, use_llm=args.llm, verbose=not args.quiet)
    requests = load_requests(Path(args.input))
    context = all_requests(dataset_dir)
    known = {r.case_id for r in context}
    context += [r for r in requests if r.case_id not in known]
    results = pipe.run(requests, context_requests=context)

    problems = 0
    for res in results:
        errs = validate_steps(to_json_list(res.steps), res.status)
        if errs:
            problems += 1
            print(f"[warn] {res.case_id}: {errs}", file=sys.stderr)
    write_output(results, Path(args.output))
    if args.trace:
        with open(args.trace, "w", encoding="utf-8") as fh:
            for res in results:
                fh.write(json.dumps({"case_id": res.case_id, "status": res.status, "trace": res.trace},
                                    default=str) + "\n")
    if Path(args.input).resolve() == (dataset_dir / "deletion_requests.csv").resolve():
        from intent import classify_rules
        from usage import write_usage_report

        texts = sorted({r.request_text for r in requests if classify_rules(r.request_text).confidence < 0.9})
        docs = [ef.path for ef in pipe.index.files.values()]
        write_usage_report(
            n_requests=len(requests), elapsed_s=time.time() - t0, llm=pipe.llm, low_conf_texts=texts,
            image_paths=[p for p in docs if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
            ocr_docs=sum(1 for p in docs if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
            pdf_docs=sum(1 for p in docs if p.suffix.lower() == ".pdf"), output_path=Path(args.output),
        )

    if not args.quiet:
        counts = {}
        for r in results:
            counts[r.status] = counts.get(r.status, 0) + 1
        print(f"[done] {len(results)} cases -> {args.output}  status={counts}  "
              f"contract_violations={problems}  elapsed={time.time() - t0:.1f}s")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
