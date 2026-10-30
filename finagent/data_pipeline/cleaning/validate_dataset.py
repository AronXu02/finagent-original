from __future__ import annotations

import argparse
from pathlib import Path

from finagent.data_pipeline.cleaning.trajectory import validate_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate FinAgent ShareGPT SFT JSONL without generating data.")
    parser.add_argument("input", type=Path, help="Path to finagent_tool_calling_30k.jsonl or a sample JSONL file.")
    args = parser.parse_args(argv)

    report = validate_jsonl(args.input)
    print(report.model_dump_json(indent=2))
    return 0 if report.invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

