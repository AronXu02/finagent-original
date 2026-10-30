from __future__ import annotations

import asyncio

from finagent.packages.common.config import get_settings
from finagent.packages.orchestrator.temporal import FinAgentChatWorkflow, TEMPORAL_ACTIVITIES


def main() -> None:
    asyncio.run(_run())


async def _run() -> None:
    settings = get_settings()
    settings.validate_production_dependencies()
    if not settings.temporal_target_host:
        raise RuntimeError("必须配置 FINAGENT_TEMPORAL_TARGET_HOST")
    try:
        from temporalio.client import Client
        from temporalio.worker import UnsandboxedWorkflowRunner, Worker
    except ImportError as exc:
        raise RuntimeError(
            "Temporal Worker 需要安装 temporalio。请使用 production extra 安装 finagent。"
        ) from exc
    client = await _connect_temporal(Client, settings.temporal_target_host, settings.temporal_namespace)
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[FinAgentChatWorkflow],
        activities=TEMPORAL_ACTIVITIES,
        workflow_runner=UnsandboxedWorkflowRunner(),
    )
    print(f"finagent Temporal worker listening on task queue {settings.temporal_task_queue}")
    await worker.run()


async def _connect_temporal(client_cls, target_host: str, namespace: str):
    delay = 1.0
    for attempt in range(1, 31):
        try:
            return await client_cls.connect(target_host, namespace=namespace)
        except Exception:
            if attempt == 30:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)


if __name__ == "__main__":
    main()
