from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("finagent.apps.api.main:create_app", host="127.0.0.1", port=8000, reload=False, factory=True)


if __name__ == "__main__":
    main()
