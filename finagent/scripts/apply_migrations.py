from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from finagent.packages.common.config import get_settings


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "infra" / "migrations"


def main() -> None:
    settings = get_settings()
    settings.validate_storage_configuration()
    _apply_postgres_migrations(settings.database_url)
    _apply_neo4j_migrations(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    print("FinAgent 生产迁移已应用")


def _apply_postgres_migrations(database_url: str | None) -> None:
    if not database_url:
        raise RuntimeError("必须配置 FINAGENT_DATABASE_URL")
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("PostgreSQL 迁移需要 psycopg，请安装 production 依赖。") from exc

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        raise RuntimeError(f"未在 {MIGRATIONS_DIR} 找到 PostgreSQL 迁移文件")

    conn = _retry(lambda: psycopg.connect(database_url), "PostgreSQL")
    try:
        with conn:
            with conn.cursor() as cur:
                for path in sql_files:
                    cur.execute(path.read_text(encoding="utf-8"))
                    print(f"已应用 PostgreSQL 迁移 {path.name}")
    finally:
        conn.close()


def _apply_neo4j_migrations(
    *,
    uri: str | None,
    user: str | None,
    password: str | None,
    database: str,
) -> None:
    if not uri or not user or not password:
        raise RuntimeError("必须配置 FINAGENT_NEO4J_URI、FINAGENT_NEO4J_USER 和 FINAGENT_NEO4J_PASSWORD")
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError("Neo4j 迁移需要 neo4j，请安装 production 依赖。") from exc

    cypher_files = sorted(MIGRATIONS_DIR.glob("*.cypher"))
    if not cypher_files:
        return

    driver = _retry(lambda: GraphDatabase.driver(uri, auth=(user, password)), "Neo4j")
    try:
        _retry(lambda: driver.verify_connectivity(), "Neo4j")
        with driver.session(database=database) as session:
            for path in cypher_files:
                for statement in _split_cypher(path.read_text(encoding="utf-8")):
                    session.run(statement).consume()
                print(f"已应用 Neo4j 迁移 {path.name}")
    finally:
        driver.close()


def _retry(fn, label: str, *, attempts: int = 60, delay_seconds: float = 2.0) -> Any:
    last_exc: Exception | None = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            time.sleep(delay_seconds)
    raise RuntimeError(f"{label} 在 {attempts} 次尝试后仍未就绪") from last_exc


def _split_cypher(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


if __name__ == "__main__":
    main()
