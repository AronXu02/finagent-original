from __future__ import annotations

from finagent.packages.common.config import get_settings
from finagent.packages.common.store import build_store


def main() -> None:
    settings = get_settings()
    store = build_store(settings)
    try:
        store.init_schema()
    finally:
        store.close()
    print(f"initialized {settings.database_backend} database at {settings.database_url}")


if __name__ == "__main__":
    main()
