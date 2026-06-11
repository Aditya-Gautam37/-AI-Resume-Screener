"""
Vector Store Module
Uses ChromaDB for persistent vector storage and similarity search
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional
from loguru import logger

class VectorStore:
    """ChromaDB vector store for resume embeddings"""
    
    def __init__(self, persist_directory: str = "./data/embeddings/chroma_db"):
        self.persist_directory = persist_directory
        
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            chroma_db_impl="duckdb+parquet",
            anonymized_telemetry=False
        ))
        
        self.collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Vector store ready. Collection size: {self.collection.count()}")
    
    def add_resume(self, resume_id: str, embedding: np.ndarray, metadata: Dict):
        """Add a single resume to the vector store"""
        try:
            self.collection.add(
                embeddings=[embedding.tolist()],
                ids=[resume_id],
                metadatas=[metadata]
            )
            self.client.persist()
            logger.debug(f"Added resume: {resume_id}")
        except Exception as e:
            logger.error(f"Failed to add resume {resume_id}: {e}")
    
    def add_batch_resumes(self, resume_ids: List[str], embeddings: List[np.ndarray], 
                          metadatas: List[Dict]):
        """Add multiple resumes in batch"""
        try:
            self.collection.add(
                embeddings=[emb.tolist() for emb in embeddings],
                ids=resume_ids,
                metadatas=metadatas
            )
            self.client.persist()
            logger.info(f"Added batch of {len(resume_ids)} resumes")
        except Exception as e:
            logger.error(f"Failed to add batch: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10) -> Dict:
        """Search for similar resumes"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"ids": [[]], "distances": [[]], "metadatas": [[]]}
    
    def get_count(self) -> int:
        """Get total number of resumes in store"""
        return self.collection.count()
    
    def delete_resume(self, resume_id: str):
        """Delete a resume from the store"""
        try:
            self.collection.delete(ids=[resume_id])
            self.client.persist()
            logger.info(f"Deleted resume: {resume_id}")
        except Exception as e:
            logger.error(f"Failed to delete {resume_id}: {e}")