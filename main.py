"""Run the Hack-n-Pitch deck pipeline over a folder of PDFs."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from tqdm import tqdm

import config
from pipeline.graph import build_graph


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hack-n-Pitch PDF judging pipeline.")
    parser.add_argument("--deck", type=Path, help="Run one PDF instead of every PDF in the input directory.")
    parser.add_argument("--resume", action="store_true", help="Resume the saved batch checkpoint, if one exists.")
    parser.add_argument("--input-dir", type=Path, default=config.INPUT_PDFS_DIR, help="Directory containing PDF decks.")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Directory for summary.csv and details.csv.")
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_PATH, help="SQLite checkpoint file.")
    return parser.parse_args()


def _validate_keys() -> list[str]:
    return [name for name in ("GEMINI_API_KEY", "GROQ_API_KEY") if not os.getenv(name)]


def _find_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.deck:
        if not args.deck.is_file() or args.deck.suffix.lower() != ".pdf":
            raise ValueError(f"--deck must name an existing PDF: {args.deck}")
        return [args.deck]
    if not args.input_dir.exists():
        raise ValueError(f"Input folder does not exist: {args.input_dir}. Create it and add PDF decks.")
    pdfs = sorted(path for path in args.input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")
    if not pdfs:
        raise ValueError(f"No PDFs found in {args.input_dir}.")
    return pdfs


def _initial_state(pdfs: list[Path], output_dir: Path) -> dict:
    return {
        "decks": [{"pdf_path": str(pdf.resolve()), "deck_id": pdf.stem, "deck_name": pdf.name} for pdf in pdfs],
        "output_dir": str(output_dir.resolve()),
        "batch_errors": [],
    }


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    missing = _validate_keys()
    if missing:
        print("Missing API key(s): " + ", ".join(missing) + ". Add them to .env (see .env.example).", file=sys.stderr)
        return 2
    try:
        pdfs = _find_pdfs(args)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    thread_id = "batch-" + str(uuid5(NAMESPACE_URL, "|".join(str(pdf.resolve()) for pdf in pdfs)))
    run_config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string(str(args.checkpoint)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(run_config)
        if args.resume and snapshot.values:
            payload = None
            print("Resuming from the last SQLite checkpoint.")
        else:
            payload = _initial_state(pdfs, args.output_dir)
        try:
            # LangGraph emits one item per completed node, making progress visible
            # even for a large batch and retaining checkpoints between every stage.
            with tqdm(total=9, desc="Pipeline stages", unit="stage") as progress:
                for _ in graph.stream(payload, run_config, stream_mode="updates"):
                    progress.update(1)
            final = graph.get_state(run_config).values
        except KeyboardInterrupt:
            print("Interrupted. Re-run with --resume to continue from the last checkpoint.", file=sys.stderr)
            return 130
        except Exception as exc:
            print(f"Pipeline failed: {exc}. Re-run with --resume after fixing the error.", file=sys.stderr)
            return 1

    for name, path in final.get("export_paths", {}).items():
        print(f"{name}: {path}")
    for error in final.get("batch_errors", []):
        print(f"Warning: {error}", file=sys.stderr)
    return 0 if final.get("export_paths") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(run(_arguments()))
