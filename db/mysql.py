import pymysql
from pymysql.cursors import DictCursor
from typing import Any, Dict, List, Optional, Tuple

from db.rdb import BaseRDB

class MySQLDB(BaseRDB):
    """MySQL database wrapper using pymysql."""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
        autocommit: bool = True,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.autocommit = autocommit
        self._conn = None

    def connect(self) -> None:
        self._conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset,
            autocommit=self.autocommit,
            cursorclass=DictCursor,
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: Optional[Tuple] = None) -> int:
        with self._conn.cursor() as cursor:
            return cursor.execute(sql, params)

    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        with self._conn.cursor() as cursor:
            return cursor.executemany(sql, params_list)

    def fetchone(
        self, sql: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def fetchall(
        self, sql: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        with self._conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    def _placeholder(self, index: int) -> str:
        return "%s"

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()
