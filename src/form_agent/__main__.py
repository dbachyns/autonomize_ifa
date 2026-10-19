"""Command line: python -m form_agent <command> ...

    extract  PATH...            read forms with a reader and cache the records
    index                       rebuild the SQLite index from the cache
    eval     {tx,funsd}         score a reader's cached output against ground truth
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from form_agent.config import settings


def cmd_extract(args: argparse.Namespace) -> int:
    from form_agent.ingest import doc_id_for, find_documents
    from form_agent.readers import get_reader
    from form_agent.store import record_path, save_record
    from form_agent.validate import validate

    paths: list[Path] = []
    for p in map(Path, args.paths):
        paths += find_documents(p) if p.is_dir() else [p]
    if args.limit:
        paths = paths[: args.limit]
    kwargs = {"model": args.model} if args.model else {}
    if args.reader == "hybrid":
        kwargs["cache_dir"] = args.cache
    reader = get_reader(args.reader, **kwargs)
    todo = [p for p in paths if args.force or not record_path(args.cache, doc_id_for(p), args.reader).exists()]
    print(f"{len(paths)} documents, {len(paths) - len(todo)} already cached, extracting {len(todo)} with {args.reader}")

    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(reader.read, p): p for p in todo}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # keep going; report at the end
                failures += 1
                print(f"  FAILED {path.name}: {exc}", file=sys.stderr)
                continue
            rec.flags = [*rec.flags, *validate(rec)]
            save_record(args.cache, rec)
            n_err = sum(f.severity == "error" for f in rec.flags)
            cost = f"${rec.run.cost_usd:.4f}" if rec.run.cost_usd is not None else "-"
            print(f"  {rec.doc_id[:40]:<40} {rec.form_type:<14} {rec.run.seconds:5.1f}s {cost:>8}  "
                  f"flags={len(rec.flags)} (errors={n_err})")
    return 1 if failures else 0


def cmd_index(args: argparse.Namespace) -> int:
    from form_agent.store import build_index

    counts = build_index(args.cache, args.db)
    print(f"built {args.db}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from form_agent.evaluate import evaluate_funsd, evaluate_tx, format_report

    if args.dataset == "tx":
        report = evaluate_tx(args.ground_truth, args.cache, args.reader)
    else:
        report = evaluate_funsd(args.funsd, args.cache, args.reader)
    print(format_report(report))
    out = Path("reports") / f"eval_{report.dataset}_{args.reader}_{report.model or 'na'}.json"
    print(f"\nfull report: {report.save(out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="form_agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="read forms and cache structured records")
    p.add_argument("paths", nargs="+", help="files or directories")
    p.add_argument("--reader", default=settings.reader)
    p.add_argument("--model", help="override the reader's default model")
    p.add_argument("--cache", type=Path, default=settings.cache_dir)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N documents")
    p.add_argument("--force", action="store_true", help="re-extract even if cached")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("index", help="rebuild the SQLite index from the cache")
    p.add_argument("--cache", type=Path, default=settings.cache_dir)
    p.add_argument("--db", type=Path, default=settings.db_path)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("eval", help="score cached reader output against ground truth")
    p.add_argument("dataset", choices=["tx", "funsd"])
    p.add_argument("--reader", default=settings.reader)
    p.add_argument("--cache", type=Path, default=settings.cache_dir)
    p.add_argument("--ground-truth", type=Path, default=settings.data_dir / "ground_truth")
    p.add_argument("--funsd", type=Path, default=settings.data_dir / "external/funsd/dataset/testing_data")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
