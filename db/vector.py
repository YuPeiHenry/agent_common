from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema as MilvusFieldSchema,
    connections,
    utility,
)


class VectorDBType(Enum):
    MILVUS = "milvus"
    FAISS = "faiss"


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
