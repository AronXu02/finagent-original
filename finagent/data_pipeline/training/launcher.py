from __future__ import annotations

import argparse

from finagent.data_pipeline.training.assets import build_training_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the FinAgent LLaMA-Factory SFT training plan without launching training."
    )
    parser.add_argument("--require-dataset", action="store_true", help="Also require the JSONL dataset file.")
    parser.add_argument("--executable", default="llamafactory-cli", help="LLaMA-Factory CLI executable name.")
    args = parser.parse_args(argv)

    plan = build_training_plan(require_dataset=args.require_dataset, executable=args.executable)
    print(plan.model_dump_json(indent=2))
    return 0 if not plan.missing_assets else 1


if __name__ == "__main__":
    raise SystemExit(main())

