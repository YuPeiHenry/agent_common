from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

class BaseRDB(ABC):
    """Abstract base class for relational database operations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        raise NotImplementedError()

    @abstractmethod
    def execute(
        self, sql: str, params: Optional[Tuple] = None
    ) -> int:
        """Execute a SQL statement and return affected row count."""
        raise NotImplementedError()

    @abstractmethod
    def fetchone(
        self, sql: str, params: Optional[Tuple] = None
    ) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one row."""
        raise NotImplementedError()

    @abstractmethod
    def fetchall(
        self, sql: str, params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute query and fetch all rows."""
        raise NotImplementedError()

    def insert(
        self,
        table: str,
        data: Dict[str, Any],
        returning: Optional[str] = None,
    ) -> Union[int, Any]:
        """Insert a row into the table.

        Args:
            table: Table name.
            data: Column-value pairs to insert.
            returning: Column name to return (PostgreSQL only).

        Returns:
            Affected row count, or the returning column value if specified.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join([self._placeholder(i) for i in range(len(data))])
        values = tuple(data.values())

        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        if returning:
            sql += f" RETURNING {returning}"
            result = self.fetchone(sql, values)
            return result[returning] if result else None

        return self.execute(sql, values)

    def insert_many(
        self,
        table: str,
        columns: List[str],
        rows: List[Tuple],
    ) -> int:
        """Insert multiple rows into the table.

        Args:
            table: Table name.
            columns: List of column names.
            rows: List of value tuples.

        Returns:
            Affected row count.
        """
        if not rows:
            return 0

        cols = ", ".join(columns)
        placeholders = ", ".join([self._placeholder(i) for i in range(len(columns))])
        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

        return self.execute_many(sql, rows)

    @abstractmethod
    def execute_many(self, sql: str, params_list: List[Tuple]) -> int:
        """Execute a SQL statement with multiple parameter sets."""
        raise NotImplementedError()

    def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: str,
        where_params: Optional[Tuple] = None,
    ) -> int:
        """Update rows in the table.

        Args:
            table: Table name.
            data: Column-value pairs to update.
            where: WHERE clause (without 'WHERE' keyword).
            where_params: Parameters for the WHERE clause.

        Returns:
            Affected row count.
        """
        set_clause = ", ".join(
            [f"{k} = {self._placeholder(i)}" for i, k in enumerate(data.keys())]
        )
        values = tuple(data.values())

        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

        if where_params:
            values = values + where_params

        return self.execute(sql, values)

    def delete(
        self,
        table: str,
        where: str,
        where_params: Optional[Tuple] = None,
    ) -> int:
        """Delete rows from the table.

        Args:
            table: Table name.
            where: WHERE clause (without 'WHERE' keyword).
            where_params: Parameters for the WHERE clause.

        Returns:
            Affected row count.
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.execute(sql, where_params)

    def select(
        self,
        table: str,
        columns: List[str] = None,
        where: Optional[str] = None,
        where_params: Optional[Tuple] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Select rows from the table.

        Args:
            table: Table name.
            columns: List of columns to select (None for all).
            where: WHERE clause (without 'WHERE' keyword).
            where_params: Parameters for the WHERE clause.
            order_by: ORDER BY clause (without 'ORDER BY' keyword).
            limit: Maximum number of rows to return.
            offset: Number of rows to skip.

        Returns:
            List of rows as dictionaries.
        """
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table}"

        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

        return self.fetchall(sql, where_params)

    def select_one(
        self,
        table: str,
        columns: List[str] = None,
        where: Optional[str] = None,
        where_params: Optional[Tuple] = None,
    ) -> Optional[Dict[str, Any]]:
        """Select a single row from the table."""
        cols = ", ".join(columns) if columns else "*"
        sql = f"SELECT {cols} FROM {table}"

        if where:
            sql += f" WHERE {where}"

        return self.fetchone(sql, where_params)

    @abstractmethod
    def _placeholder(self, index: int) -> str:
        """Return the placeholder for parameterized queries."""
        raise NotImplementedError()

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        try:
            yield
            self.commit()
        except Exception:
            self.rollback()
            raise

    @abstractmethod
    def commit(self) -> None:
        """Commit the current transaction."""
        raise NotImplementedError()

    @abstractmethod
    def rollback(self) -> None:
        """Rollback the current transaction."""
        raise NotImplementedError()
