"""SQLAlchemy 连接与会话管理。"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

import config_data as config


DATABASE_URL = (
    f"mysql+pymysql://{config.mysql_user}:{config.mysql_password}"
    f"@{config.mysql_host}:{config.mysql_port}/{config.mysql_database}?charset=utf8mb4"
)
SERVER_DATABASE_URL = (
    f"mysql+pymysql://{config.mysql_user}:{config.mysql_password}"
    f"@{config.mysql_host}:{config.mysql_port}/mysql?charset=utf8mb4"
)

Base = declarative_base()

server_engine = create_engine(SERVER_DATABASE_URL, pool_pre_ping=True)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def ensure_database_exists() -> None:
    """确保 MySQL 数据库存在。"""
    with server_engine.begin() as connection:
        connection.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{config.mysql_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )


@contextmanager
def session_scope():
    """统一管理 SQLAlchemy 会话的提交与回滚。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
