from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finagent.packages.common.config import get_settings
from finagent.packages.common.store import build_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入租户范围内的组合快照。")
    parser.add_argument("--tenant-id", required=True, help="租户 ID。")
    parser.add_argument("--user-id", required=True, help="用户 ID。")
    parser.add_argument("--account-id", default="default", help="账户 ID。")
    parser.add_argument("--currency", default="USD", help="组合币种。")
    parser.add_argument("--snapshot", required=True, type=Path, help="组合快照 JSON 文件路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    positions: list[dict[str, Any]] = payload["positions"]
    store = build_store(get_settings())
    try:
        store.upsert_portfolio_snapshot(
            args.tenant_id,
            args.user_id,
            account_id=args.account_id,
            currency=payload.get("currency", args.currency),
            snapshot_at=payload.get("snapshot_at"),
            positions=positions,
        )
    finally:
        store.close()
    print(f"已为 {args.tenant_id}/{args.user_id}/{args.account_id} 导入 {len(positions)} 条持仓")


if __name__ == "__main__":
    main()
