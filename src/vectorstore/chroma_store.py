"""
ChromaDB Persistent Vector Store Module.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import time
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.vectorstore.embedding_engine import EmbeddingEngine, DEFAULT_MODEL_NAME

DEFAULT_COLLECTION_NAME = "ragops_enterprise"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DB_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "chroma_db")

class ChromaVectorStore:
    """Manages persistent ChromaDB vector index with full metadata preservation."""

    def __init__(self, 
                 db_dir: str = DEFAULT_DB_DIR, 
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 embedding_model_name: str = DEFAULT_MODEL_NAME):
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model_name)
        
        try:
            self.client = chromadb.PersistentClient(path=self.db_dir)
        except BaseException:
            import shutil
            shutil.rmtree(self.db_dir, ignore_errors=True)
            os.makedirs(self.db_dir, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.db_dir)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedding_model_name}
        )

    def reload_index(self):
        """Reloads persistent collection from disk."""
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_engine.model_name}
        )

    def reset_collection(self):
        """Clears existing items in the collection for a fresh rebuild."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_engine.model_name}
        )

    def add_chunks(self, chunk_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Indexes chunk records into ChromaDB with deterministic IDs and complete metadata."""
        if not chunk_records:
            return {"added_count": 0, "indexing_time": 0.0}

        start_time = time.time()
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunk_records:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            meta = dict(chunk["metadata"])
            
            # Embed document_id into metadata top level for filtering
            meta["document_id"] = chunk["document_id"]
            meta["chunk_index"] = chunk["chunk_index"]
            
            ids.append(chunk_id)
            documents.append(text)
            metadatas.append(meta)
            
        embeddings = self.embedding_engine.embed_documents(documents)
        
        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        elapsed = time.time() - start_time
        return {
            "added_count": len(ids),
            "indexing_time": elapsed,
            "total_items_in_collection": self.collection.count()
        }

    def query(self, 
              query_text: str, 
              n_results: int = 5, 
              where_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Queries the vector index using cosine similarity and optional metadata filters."""
        if not query_text or not query_text.strip():
            return []

        query_embedding = self.embedding_engine.embed_text(query_text)
        
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_filter:
            kwargs["where"] = where_filter
            
        try:
            results = self.collection.query(**kwargs)
        except Exception:
            self.reload_index()
            try:
                results = self.collection.query(**kwargs)
            except Exception:
                return []
        
        formatted = []
        if results and results["ids"] and len(results["ids"]) > 0:
            res_ids = results["ids"][0]
            res_docs = results["documents"][0]
            res_metas = results["metadatas"][0]
            res_dists = results["distances"][0]
            
            for i in range(len(res_ids)):
                # Convert distance to similarity score (cosine distance in Chroma = 1 - cosine_sim)
                dist = res_dists[i]
                sim_score = round(1.0 - dist, 4) if dist is not None else 0.0
                
                formatted.append({
                    "chunk_id": res_ids[i],
                    "text": res_docs[i],
                    "metadata": res_metas[i],
                    "distance": dist,
                    "score": sim_score
                })
                
        return formatted

    def count(self) -> int:
        return self.collection.count()
