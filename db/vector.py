"""Vector database wrapper for Milvus."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema as MilvusFieldSchema,
    connections,
    utility,
)
from typing import Any, Dict, List, Optional


class VectorDBType(Enum):
    MILVUS = "milvus"


class MetricType(Enum):
    L2 = "L2"
    IP = "IP"  # Inner Product
    COSINE = "COSINE"


class IndexType(Enum):
    FLAT = "FLAT"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    AUTOINDEX = "AUTOINDEX"


@dataclass
class FieldSchema:
    """Schema definition for a collection field."""

    name: str
    dtype: str  # "int64", "varchar", "float_vector", "float", "bool", "json"
    is_primary: bool = False
    auto_id: bool = False
    max_length: Optional[int] = None  # For varchar
    dim: Optional[int] = None  # For vector fields


@dataclass
class SearchResult:
    """Single search result."""

    id: Any
    distance: float
    entity: Dict[str, Any]


class BaseVectorDB(ABC):
    """Abstract base class for vector database operations."""

    @abstractmethod
    def connect(self) -> None:
        """Establish database connection."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        raise NotImplementedError()

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        fields: List[FieldSchema],
        description: str = "",
    ) -> None:
        """Create a new collection."""
        raise NotImplementedError()

    @abstractmethod
    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        raise NotImplementedError()

    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        raise NotImplementedError()

    @abstractmethod
    def list_collections(self) -> List[str]:
        """List all collections."""
        raise NotImplementedError()

    @abstractmethod
    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: IndexType = IndexType.AUTOINDEX,
        metric_type: MetricType = MetricType.COSINE,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create an index on a vector field."""
        raise NotImplementedError()

    @abstractmethod
    def load_collection(self, collection_name: str) -> None:
        """Load a collection into memory for searching."""
        raise NotImplementedError()

    @abstractmethod
    def release_collection(self, collection_name: str) -> None:
        """Release a collection from memory."""
        raise NotImplementedError()

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Insert entities into a collection."""
        raise NotImplementedError()

    def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        ids: Optional[List[Any]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        id_field: str = "id",
        vector_field: str = "vector",
    ) -> List[Any]:
        """Convenience method to insert vectors with optional metadata.

        Args:
            collection_name: Name of the collection.
            vectors: List of vectors.
            ids: Optional list of IDs (if not auto-generated).
            metadata: Optional list of metadata dicts for each vector.
            id_field: Name of the ID field.
            vector_field: Name of the vector field.

        Returns:
            List of inserted primary keys.
        """
        data = []
        for i, vector in enumerate(vectors):
            entity = {vector_field: vector}
            if ids is not None:
                entity[id_field] = ids[i]
            if metadata is not None and i < len(metadata):
                entity.update(metadata[i])
            data.append(entity)

        return self.insert(collection_name, data)

    @abstractmethod
    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        vector_field: str = "vector",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        expr: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """Search for similar vectors."""
        raise NotImplementedError()

    @abstractmethod
    def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query entities by filter expression."""
        raise NotImplementedError()

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        expr: str,
    ) -> int:
        """Delete entities by filter expression."""
        raise NotImplementedError()

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Upsert entities into a collection."""
        raise NotImplementedError()

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Get the number of entities in a collection."""
        raise NotImplementedError()

    @abstractmethod
    def flush(self, collection_name: str) -> None:
        """Flush a collection to persist data."""
        raise NotImplementedError()


class MilvusDB(BaseVectorDB):
    """Milvus vector database wrapper using pymilvus."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        db_name: str = "default",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self._client = None

    def connect(self) -> None:
        """Establish connection to Milvus."""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db_name=self.db_name,
        )

    def close(self) -> None:
        """Close connection to Milvus."""
        connections.disconnect(alias="default")

    def create_collection(
        self,
        collection_name: str,
        fields: List[FieldSchema],
        description: str = "",
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection.
            fields: List of field schemas.
            description: Collection description.
        """
        dtype_map = {
            "int64": DataType.INT64,
            "varchar": DataType.VARCHAR,
            "float_vector": DataType.FLOAT_VECTOR,
            "float": DataType.FLOAT,
            "double": DataType.DOUBLE,
            "bool": DataType.BOOL,
            "json": DataType.JSON,
            "int32": DataType.INT32,
            "int16": DataType.INT16,
            "int8": DataType.INT8,
        }

        milvus_fields = []
        for field in fields:
            kwargs = {
                "name": field.name,
                "dtype": dtype_map[field.dtype],
                "is_primary": field.is_primary,
            }
            if field.is_primary:
                kwargs["auto_id"] = field.auto_id
            if field.max_length is not None:
                kwargs["max_length"] = field.max_length
            if field.dim is not None:
                kwargs["dim"] = field.dim

            milvus_fields.append(MilvusFieldSchema(**kwargs))

        schema = CollectionSchema(fields=milvus_fields, description=description)
        Collection(name=collection_name, schema=schema)

    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        utility.drop_collection(collection_name)

    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return utility.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        """List all collections."""
        return utility.list_collections()

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: IndexType = IndexType.AUTOINDEX,
        metric_type: MetricType = MetricType.COSINE,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create an index on a vector field.

        Args:
            collection_name: Name of the collection.
            field_name: Name of the vector field.
            index_type: Type of index.
            metric_type: Distance metric type.
            params: Additional index parameters (e.g., nlist for IVF).
        """
        collection = Collection(collection_name)
        index_params = {
            "index_type": index_type.value,
            "metric_type": metric_type.value,
            "params": params or {},
        }
        collection.create_index(field_name=field_name, index_params=index_params)

    def load_collection(self, collection_name: str) -> None:
        """Load a collection into memory for searching."""
        collection = Collection(collection_name)
        collection.load()

    def release_collection(self, collection_name: str) -> None:
        """Release a collection from memory."""
        collection = Collection(collection_name)
        collection.release()

    def insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Insert entities into a collection.

        Args:
            collection_name: Name of the collection.
            data: List of entities, each as a dict with field names as keys.

        Returns:
            List of inserted primary keys.
        """
        collection = Collection(collection_name)
        result = collection.insert(data)
        return result.primary_keys

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        vector_field: str = "vector",
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        expr: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[List[SearchResult]]:
        """Search for similar vectors.

        Args:
            collection_name: Name of the collection.
            vectors: Query vectors.
            vector_field: Name of the vector field to search.
            limit: Maximum number of results per query.
            output_fields: Fields to include in results.
            expr: Filter expression (e.g., "age > 20").
            metric_type: Override metric type for search.
            params: Search parameters (e.g., nprobe for IVF).

        Returns:
            List of search results for each query vector.
        """
        collection = Collection(collection_name)

        search_params = {"params": params or {}}
        if metric_type:
            search_params["metric_type"] = metric_type.value

        results = collection.search(
            data=vectors,
            anns_field=vector_field,
            param=search_params,
            limit=limit,
            output_fields=output_fields,
            expr=expr,
        )

        search_results = []
        for hits in results:
            hit_results = []
            for hit in hits:
                entity = {field: hit.entity.get(field) for field in (output_fields or [])}
                hit_results.append(
                    SearchResult(id=hit.id, distance=hit.distance, entity=entity)
                )
            search_results.append(hit_results)

        return search_results

    def query(
        self,
        collection_name: str,
        expr: str,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query entities by filter expression.

        Args:
            collection_name: Name of the collection.
            expr: Filter expression (e.g., "id in [1, 2, 3]").
            output_fields: Fields to include in results.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching entities.
        """
        collection = Collection(collection_name)

        kwargs = {"expr": expr}
        if output_fields:
            kwargs["output_fields"] = output_fields
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset

        return collection.query(**kwargs)

    def delete(
        self,
        collection_name: str,
        expr: str,
    ) -> int:
        """Delete entities by filter expression.

        Args:
            collection_name: Name of the collection.
            expr: Filter expression (e.g., "id in [1, 2, 3]").

        Returns:
            Number of deleted entities.
        """
        collection = Collection(collection_name)
        result = collection.delete(expr)
        return result.delete_count

    def upsert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
    ) -> List[Any]:
        """Upsert entities into a collection.

        Args:
            collection_name: Name of the collection.
            data: List of entities to upsert.

        Returns:
            List of upserted primary keys.
        """
        collection = Collection(collection_name)
        result = collection.upsert(data)
        return result.primary_keys

    def count(self, collection_name: str) -> int:
        """Get the number of entities in a collection."""
        collection = Collection(collection_name)
        return collection.num_entities

    def flush(self, collection_name: str) -> None:
        """Flush a collection to persist data."""
        collection = Collection(collection_name)
        collection.flush()


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
        host: Database host.
        port: Database port.
        user: Database user.
        password: Database password.
        db_name: Database name.
        **kwargs: Additional arguments passed to the database constructor.

    Returns:
        Vector database instance.
    """
    match db_type:
        case VectorDBType.MILVUS:
            return MilvusDB(host, port, user, password, db_name, **kwargs)
        case _:
            raise ValueError(f"Unsupported vector database type: {db_type}")
