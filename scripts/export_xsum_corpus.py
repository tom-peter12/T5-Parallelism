#!/usr/bin/env python3
import argparse
import json
import re

from datasets import load_dataset


def normalize_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XSum to Megatron JSONL format.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to export.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--include-summary",
        dest="include_summary",
        action="store_true",
        help="Include summary text in the exported corpus.",
    )
    parser.add_argument(
        "--no-include-summary",
        dest="include_summary",
        action="store_false",
        help="Exclude summary text and keep only article text.",
    )
    parser.set_defaults(include_summary=True)
    return parser.parse_args()


def split_limit(split_name: str, args: argparse.Namespace) -> int:
    if split_name == "train":
        return args.max_train_samples
    if split_name == "validation":
        return args.max_validation_samples
    if split_name == "test":
        return args.max_test_samples
    return 0


def build_text(document: str, summary: str, include_summary: bool) -> str:
    if include_summary:
        return f"summarize: {document} summary: {summary}"
    return f"summarize: {document}"


def main() -> None:
    args = parse_args()
    dataset = load_dataset("xsum")

    total_written = 0
    split_counts = {}

    with open(args.output, "w", encoding="utf-8") as handle:
        for split_name in args.splits:
            if split_name not in dataset:
                raise ValueError(f"Unknown split '{split_name}'. Available: {list(dataset.keys())}")

            limit = split_limit(split_name, args)
            written = 0

            for row_idx, row in enumerate(dataset[split_name]):
                if limit and row_idx >= limit:
                    break

                document = normalize_text(row["document"])
                summary = normalize_text(row["summary"])

                if not document:
                    continue
                if args.include_summary and not summary:
                    continue

                text = build_text(document, summary, args.include_summary)
                json.dump({"text": text}, handle, ensure_ascii=False)
                handle.write("\n")
                written += 1

            split_counts[split_name] = written
            total_written += written

    print("Export complete.")
    for split_name in args.splits:
        print(f"  {split_name}: {split_counts.get(split_name, 0)} rows")
    print(f"  total: {total_written} rows")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()