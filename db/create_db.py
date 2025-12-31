from enum import Enum

from db.rdb import BaseRDB
from db.mysql import MySQLDB
from db.psql import PostgresDB

from db.faiss import FaissDB
from db.milvus import MilvusDB
from db.vector import BaseVectorDB, VectorDBType

class DBType(Enum):
    MYSQL = "mysql"
    POSTGRES = "postgres"

def create_db(
    db_type: DBType,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
    **kwargs,
) -> BaseRDB:
    """Factory function to create a database instance.

    Args:
        db_type: Type of database (MySQL or PostgreSQL).
        host: Database host.
        port: Database port.
        user: Database user.
        password: Database password.
        database: Database name.
        **kwargs: Additional arguments passed to the database constructor.

    Returns:
        Database instance.
    """
    match(db_type):
        case DBType.MYSQL:
            return MySQLDB(host, port, user, password, database, **kwargs)
        case DBType.POSTGRES:
            return PostgresDB(host, port, user, password, database, **kwargs)
        case _:
            raise ValueError(f"Unsupported database type: {db_type}")

def create_vector_db(
    db_type: VectorDBType,
    host: str = "localhost",
    port: int = 19530,
    user: str = "",
    password: str = "",
    db_name: str = "default",
    **kwargs,
) -> BaseVectorDB:
    """Factory function to create a vector database instance.

    Args:
        db_type: Type of vector database.
        host: Database host (Milvus only).
        port: Database port (Milvus only).
        user: Database user (Milvus only).
        password: Database password (Milvus only).
        db_name: Database name (Milvus only).
        **kwargs: Additional arguments passed to the database constructor.
            For FAISS:
                - persist_directory: Directory to persist indexes to disk.
                - normalize_vectors: Whether to L2-normalize vectors.

    Returns:
        Vector database instance.
    """
    match db_type:
        case VectorDBType.MILVUS:
            return MilvusDB(host, port, user, password, db_name, **kwargs)
        case VectorDBType.FAISS:
            return FaissDB(**kwargs)
        case _:
            raise ValueError(f"Unsupported vector database type: {db_type}")
