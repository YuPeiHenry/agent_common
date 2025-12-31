import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Any, Dict, List, Optional, Tuple

from db.rdb import BaseRDB

class PostgresDB(BaseRDB):
    """PostgreSQL database wrapper using psycopg2."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        autocommit: bool = True,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.autocommit = autocommit
        self._conn = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.database,
            cursor_factory=RealDictCursor,
        )
        self._conn.autocommit = self.autocommit

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.rowcount

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        with self._conn.cursor() as cursor:
            cursor.executemany(sql, params_list)
            return cursor.rowcount

    def fetchone(
        self, sql: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetchall(
        self, sql: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def _placeholder(self, index: int) -> str:
        return "%s"

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()
