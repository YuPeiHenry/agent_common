from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from db.vector import BaseVectorDB, FieldSchema, MetricType

@dataclass
class _FaissCollection:
    """Internal representation of a FAISS collection."""

    index: faiss.Index
    dimension: int
    metric_type: MetricType
    fields: List[FieldSchema]
    description: str
    id_field: str
    vector_field: str
    ids: List[Any] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    next_auto_id: int = 0


class FaissDB(BaseVectorDB):
    """FAISS vector database wrapper.

    FAISS is an in-memory vector similarity search library. This wrapper
    provides a collection-based interface similar to Milvus, with optional
    persistence to disk.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        normalize_vectors: bool = False,
    ):
        """Initialize FAISS database.

        Args:
            persist_directory: Directory to persist indexes to disk.
                If None, data is only stored in memory.
            normalize_vectors: Whether to L2-normalize vectors before indexing.
                Useful for cosine similarity with IP metric.
        """
        self.persist_directory = persist_directory
        self.normalize_vectors = normalize_vectors
        self._collections: Dict[str, _FaissCollection] = {}
        self._connected = False

    def connect(self) -> None:
        """Initialize the database (load persisted data if available)."""
        if self.persist_directory:
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            self._load_all_collections()
        self._connected = True

    def close(self) -> None:
        """Persist all collections and close the database."""
        if self.persist_directory:
            self._save_all_collections()
        self._collections.clear()
        self._connected = False

    def _get_index_path(self, collection_name: str) -> str:
        """Get the path for a collection's index file."""
        return os.path.join(self.persist_directory, f"{collection_name}.index")

    def _get_metadata_path(self, collection_name: str) -> str:
        """Get the path for a collection's metadata file."""
        return os.path.join(self.persist_directory, f"{collection_name}.meta.json")

    def _save_collection(self, collection_name: str) -> None:
        """Save a single collection to disk."""
        if not self.persist_directory:
            return

        coll = self._collections[collection_name]
        faiss.write_index(coll.index, self._get_index_path(collection_name))

        meta = {
            "dimension": coll.dimension,
            "metric_type": coll.metric_type.value,
            "fields": [
                {
                    "name": f.name,
                    "dtype": f.dtype,
                    "is_primary": f.is_primary,
                    "auto_id": f.auto_id,
                    "max_length": f.max_length,
                    "dim": f.dim,
                }
                for f in coll.fields
            ],
            "description": coll.description,
            "id_field": coll.id_field,
            "vector_field": coll.vector_field,
            "ids": coll.ids,
            "metadata": coll.metadata,
            "next_auto_id": coll.next_auto_id,
        }
        with open(self._get_metadata_path(collection_name), "w") as f:
            json.dump(meta, f)

    def _load_collection(self, collection_name: str) -> None:
        """Load a single collection from disk."""
        index_path = self._get_index_path(collection_name)
        meta_path = self._get_metadata_path(collection_name)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return

        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)

        fields = [
            FieldSchema(
                name=field["name"],
                dtype=field["dtype"],
                is_primary=field["is_primary"],
                auto_id=field["auto_id"],
                max_length=field["max_length"],
                dim=field["dim"],
            )
            for field in meta["fields"]
        ]

        self._collections[collection_name] = _FaissCollection(
            index=index,
            dimension=meta["dimension"],
            metric_type=MetricType(meta["metric_type"]),
            fields=fields,
            description=meta["description"],
            id_field=meta["id_field"],
            vector_field=meta["vector_field"],
            ids=meta["ids"],
            metadata=meta["metadata"],
            next_auto_id=meta["next_auto_id"],
        )

    def _save_all_collections(self) -> None:
        """Save all collections to disk."""
        for name in self._collections:
            self._save_collection(name)

    def _load_all_collections(self) -> None:
        """Load all collections from disk."""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            return

        for filename in os.listdir(self.persist_directory):
            if filename.endswith(".index"):
                collection_name = filename[:-6]
                self._load_collection(collection_name)

    def create_collection(
        self,
        collection_name: str,
        fields: List[FieldSchema],
        description: str = "",
    ) -> None:
        """Create a new collection.

        Args:
            collection_name: Name of the collection.
            fields: List of field schemas. Must include exactly one primary key
                field and one float_vector field.
            description: Collection description.
        """
        if collection_name in self._collections:
            raise ValueError(f"Collection '{collection_name}' already exists")

        id_field = None
        vector_field = None
        dimension = None
        auto_id = False

        for field in fields:
            if field.is_primary:
                id_field = field.name
                auto_id = field.auto_id
            if field.dtype == "float_vector":
                vector_field = field.name
                dimension = field.dim

        if not id_field:
            raise ValueError("Collection must have a primary key field")
        if not vector_field or not dimension:
            raise ValueError("Collection must have a float_vector field with dimension")

        index = faiss.IndexFlatL2(dimension)

        self._collections[collection_name] = _FaissCollection(
            index=index,
            dimension=dimension,
            metric_type=MetricType.L2,
            fields=fields,
            description=description,
            id_field=id_field,
            vector_field=vector_field,
            ids=[],
            metadata=[],
            next_auto_id=0 if auto_id else -1,
        )

    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        del self._collections[collection_name]

        if self.persist_directory:
            index_path = self._get_index_path(collection_name)
            meta_path = self._get_metadata_path(collection_name)
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)

    def has_collection(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        return collection_name in self._collections

    def list_collections(self) -> List[str]:
        """List all collections."""
        return list(self._collections.keys())

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: IndexType = IndexType.FLAT,
        metric_type: MetricType = MetricType.L2,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create/replace the index on the vector field.

        Args:
            collection_name: Name of the collection.
            field_name: Name of the vector field (must be the collection's vector field).
            index_type: Type of index (FLAT, IVF_FLAT, IVF_PQ, HNSW).
            metric_type: Distance metric type (L2, IP, COSINE).
            params: Index parameters:
                - IVF_FLAT/IVF_PQ: {"nlist": 100}
                - IVF_PQ: {"nlist": 100, "m": 8, "nbits": 8}
                - HNSW: {"M": 32, "efConstruction": 40}
        """
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]
        if field_name != coll.vector_field:
            raise ValueError(f"Field '{field_name}' is not the vector field")

        params = params or {}
        d = coll.dimension

        if metric_type == MetricType.COSINE:
            metric_type = MetricType.IP

        if metric_type == MetricType.L2:
            if index_type == IndexType.FLAT:
                new_index = faiss.IndexFlatL2(d)
            elif index_type == IndexType.IVF_FLAT:
                nlist = params.get("nlist", 100)
                quantizer = faiss.IndexFlatL2(d)
                new_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            elif index_type == IndexType.IVF_PQ:
                nlist = params.get("nlist", 100)
                m = params.get("m", 8)
                nbits = params.get("nbits", 8)
                quantizer = faiss.IndexFlatL2(d)
                new_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
            elif index_type == IndexType.HNSW:
                M = params.get("M", 32)
                new_index = faiss.IndexHNSWFlat(d, M)
                if "efConstruction" in params:
                    new_index.hnsw.efConstruction = params["efConstruction"]
            else:
                new_index = faiss.IndexFlatL2(d)
        else:  # IP
            if index_type == IndexType.FLAT:
                new_index = faiss.IndexFlatIP(d)
            elif index_type == IndexType.IVF_FLAT:
                nlist = params.get("nlist", 100)
                quantizer = faiss.IndexFlatIP(d)
                new_index = faiss.IndexIVFFlat(
                    quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
                )
            elif index_type == IndexType.IVF_PQ:
                nlist = params.get("nlist", 100)
                m = params.get("m", 8)
                nbits = params.get("nbits", 8)
                quantizer = faiss.IndexFlatIP(d)
                new_index = faiss.IndexIVFPQ(
                    quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT
                )
            elif index_type == IndexType.HNSW:
                M = params.get("M", 32)
                new_index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
                if "efConstruction" in params:
                    new_index.hnsw.efConstruction = params["efConstruction"]
            else:
                new_index = faiss.IndexFlatIP(d)

        if coll.index.ntotal > 0:
            vectors = faiss.rev_swig_ptr(
                coll.index.get_xb(), coll.index.ntotal * d
            ).reshape(coll.index.ntotal, d)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)

            if hasattr(new_index, "train") and not new_index.is_trained:
                new_index.train(vectors)
            new_index.add(vectors)

        coll.index = new_index
        coll.metric_type = metric_type

    def load_collection(self, collection_name: str) -> None:
        """Load a collection (no-op for FAISS as data is always in memory)."""
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

    def release_collection(self, collection_name: str) -> None:
        """Release a collection (no-op for FAISS)."""
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

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
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]
        inserted_ids = []

        vectors = []
        for entity in data:
            if coll.vector_field not in entity:
                raise ValueError(f"Missing vector field '{coll.vector_field}'")

            vector = entity[coll.vector_field]
            if len(vector) != coll.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {coll.dimension}, "
                    f"got {len(vector)}"
                )
            vectors.append(vector)

            if coll.next_auto_id >= 0:
                entity_id = coll.next_auto_id
                coll.next_auto_id += 1
            else:
                if coll.id_field not in entity:
                    raise ValueError(f"Missing ID field '{coll.id_field}'")
                entity_id = entity[coll.id_field]

            inserted_ids.append(entity_id)
            coll.ids.append(entity_id)

            meta = {k: v for k, v in entity.items() if k != coll.vector_field}
            meta[coll.id_field] = entity_id
            coll.metadata.append(meta)

        vectors_np = np.array(vectors, dtype=np.float32)
        if self.normalize_vectors:
            faiss.normalize_L2(vectors_np)

        if hasattr(coll.index, "train") and not coll.index.is_trained:
            coll.index.train(vectors_np)

        coll.index.add(vectors_np)

        return inserted_ids

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
            vector_field: Name of the vector field (unused, for API compatibility).
            limit: Maximum number of results per query.
            output_fields: Fields to include in results.
            expr: Filter expression (basic support: "field == value", "field in [...]").
            metric_type: Override metric type (unused, uses index's metric).
            params: Search parameters:
                - IVF indexes: {"nprobe": 10}
                - HNSW: {"efSearch": 40}

        Returns:
            List of search results for each query vector.
        """
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]
        params = params or {}

        if hasattr(coll.index, "nprobe") and "nprobe" in params:
            coll.index.nprobe = params["nprobe"]
        if hasattr(coll.index, "hnsw") and "efSearch" in params:
            coll.index.hnsw.efSearch = params["efSearch"]

        query_vectors = np.array(vectors, dtype=np.float32)
        if self.normalize_vectors:
            faiss.normalize_L2(query_vectors)

        k = min(limit, coll.index.ntotal) if coll.index.ntotal > 0 else limit
        if k == 0:
            return [[] for _ in vectors]

        distances, indices = coll.index.search(query_vectors, k)

        filter_indices = None
        if expr:
            filter_indices = self._evaluate_filter(coll, expr)

        results = []
        for i in range(len(vectors)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                if idx < 0 or idx >= len(coll.ids):
                    continue

                if filter_indices is not None and idx not in filter_indices:
                    continue

                entity_id = coll.ids[idx]
                distance = float(distances[i][j])
                meta = coll.metadata[idx]

                entity = {}
                if output_fields:
                    for field in output_fields:
                        if field in meta:
                            entity[field] = meta[field]

                query_results.append(
                    SearchResult(id=entity_id, distance=distance, entity=entity)
                )

                if len(query_results) >= limit:
                    break

            results.append(query_results)

        return results

    def _evaluate_filter(
        self, coll: _FaissCollection, expr: str
    ) -> set:
        """Evaluate a simple filter expression and return matching indices.

        Supports basic expressions:
        - "field == value"
        - "field != value"
        - "field in [value1, value2]"
        - "field > value", "field >= value", "field < value", "field <= value"
        """
        matching_indices = set()

        expr = expr.strip()

        for i, meta in enumerate(coll.metadata):
            if self._matches_expr(meta, expr):
                matching_indices.add(i)

        return matching_indices

    def _matches_expr(self, meta: Dict[str, Any], expr: str) -> bool:
        """Check if metadata matches the expression."""
        expr = expr.strip()

        if " and " in expr.lower():
            parts = expr.lower().split(" and ")
            idx = 0
            sub_exprs = []
            current = ""
            for char in expr:
                current += char
                if current.lower().endswith(" and "):
                    sub_exprs.append(current[:-5].strip())
                    current = ""
            if current:
                sub_exprs.append(current.strip())
            return all(self._matches_expr(meta, sub) for sub in sub_exprs)

        if " or " in expr.lower():
            parts = expr.lower().split(" or ")
            idx = 0
            sub_exprs = []
            current = ""
            for char in expr:
                current += char
                if current.lower().endswith(" or "):
                    sub_exprs.append(current[:-4].strip())
                    current = ""
            if current:
                sub_exprs.append(current.strip())
            return any(self._matches_expr(meta, sub) for sub in sub_exprs)

        if " in " in expr:
            field, rest = expr.split(" in ", 1)
            field = field.strip()
            values_str = rest.strip()
            if values_str.startswith("[") and values_str.endswith("]"):
                values_str = values_str[1:-1]
            values = [v.strip().strip("'\"") for v in values_str.split(",")]
            field_value = meta.get(field)
            return str(field_value) in values or field_value in values

        for op in ["==", "!=", ">=", "<=", ">", "<"]:
            if op in expr:
                field, value = expr.split(op, 1)
                field = field.strip()
                value = value.strip().strip("'\"")

                field_value = meta.get(field)
                if field_value is None:
                    return False

                try:
                    if isinstance(field_value, (int, float)):
                        value = type(field_value)(value)
                except (ValueError, TypeError):
                    pass

                if op == "==":
                    return field_value == value or str(field_value) == value
                elif op == "!=":
                    return field_value != value and str(field_value) != value
                elif op == ">":
                    return field_value > value
                elif op == ">=":
                    return field_value >= value
                elif op == "<":
                    return field_value < value
                elif op == "<=":
                    return field_value <= value

        return True

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
            expr: Filter expression.
            output_fields: Fields to include in results.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching entities.
        """
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]
        matching_indices = self._evaluate_filter(coll, expr)

        results = []
        sorted_indices = sorted(matching_indices)

        start = offset or 0
        end = (start + limit) if limit else len(sorted_indices)

        for idx in sorted_indices[start:end]:
            meta = coll.metadata[idx]
            if output_fields:
                entity = {k: v for k, v in meta.items() if k in output_fields}
            else:
                entity = dict(meta)
            results.append(entity)

        return results

    def delete(
        self,
        collection_name: str,
        expr: str,
    ) -> int:
        """Delete entities by filter expression.

        Note: FAISS does not support efficient deletion. This rebuilds the index
        without the deleted vectors.

        Args:
            collection_name: Name of the collection.
            expr: Filter expression.

        Returns:
            Number of deleted entities.
        """
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]
        delete_indices = self._evaluate_filter(coll, expr)

        if not delete_indices:
            return 0

        keep_indices = [i for i in range(len(coll.ids)) if i not in delete_indices]

        if not keep_indices:
            coll.index.reset()
            coll.ids.clear()
            coll.metadata.clear()
            return len(delete_indices)

        old_vectors = faiss.rev_swig_ptr(
            coll.index.get_xb(), coll.index.ntotal * coll.dimension
        ).reshape(coll.index.ntotal, coll.dimension)

        new_vectors = np.ascontiguousarray(
            old_vectors[keep_indices], dtype=np.float32
        )
        new_ids = [coll.ids[i] for i in keep_indices]
        new_metadata = [coll.metadata[i] for i in keep_indices]

        coll.index.reset()
        if hasattr(coll.index, "train") and not coll.index.is_trained:
            coll.index.train(new_vectors)
        coll.index.add(new_vectors)

        coll.ids = new_ids
        coll.metadata = new_metadata

        return len(delete_indices)

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
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        coll = self._collections[collection_name]

        ids_to_delete = []
        for entity in data:
            if coll.id_field in entity:
                ids_to_delete.append(entity[coll.id_field])

        if ids_to_delete:
            id_list = ", ".join(repr(id_) for id_ in ids_to_delete)
            self.delete(collection_name, f"{coll.id_field} in [{id_list}]")

        return self.insert(collection_name, data)

    def count(self, collection_name: str) -> int:
        """Get the number of entities in a collection."""
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        return self._collections[collection_name].index.ntotal

    def flush(self, collection_name: str) -> None:
        """Flush a collection to persist data."""
        if collection_name not in self._collections:
            raise ValueError(f"Collection '{collection_name}' does not exist")

        if self.persist_directory:
            self._save_collection(collection_name)
