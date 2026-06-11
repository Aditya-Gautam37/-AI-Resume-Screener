"""
FAISS Vector Store for Resume Screening
Fast similarity search with Facebook AI Similarity Search
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger

class FAISSVectorStore:
    """FAISS-based vector store for resume embeddings"""
    
    def __init__(self, dimension: int = 384, persist_path: str = "./data/embeddings/faiss_index"):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            persist_path: Path to save/load index
        """
        self.dimension = dimension
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # Create FAISS index (IVF for faster search with large datasets)
        self.index = None
        self.resume_ids = []  # Store IDs corresponding to index positions
        self.resume_metadata = []  # Store metadata
        
        self._load_or_create_index()
        logger.info(f"FAISS store initialized with dimension {dimension}")
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        index_path = self.persist_path / "faiss.index"
        metadata_path = self.persist_path / "metadata.pkl"
        
        if index_path.exists() and metadata_path.exists():
            try:
                # Load existing index
                self.index = faiss.read_index(str(index_path))
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.resume_ids = metadata['ids']
                    self.resume_metadata = metadata['metadata']
                logger.info(f"Loaded existing index with {len(self.resume_ids)} resumes")
            except Exception as e:
                logger.error(f"Failed to load index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Use IVF (Inverted File) index for faster search with large datasets
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        
        # Need to train the index when we have data
        self.index_initialized = False
        logger.info("Created new FAISS index (needs training)")
    
    def _train_index(self, embeddings: np.ndarray):
        """Train the IVF index with embeddings"""
        if embeddings.shape[0] >= 100:  # Need at least 100 vectors to train
            logger.info(f"Training index with {embeddings.shape[0]} vectors")
            self.index.train(embeddings)
            self.index_initialized = True
        else:
            logger.warning(f"Not enough vectors to train (need 100, have {embeddings.shape[0]})")
            self.index_initialized = False
    
    def add_resumes(self, resume_ids: List[str], embeddings: np.ndarray, 
                    metadatas: List[Dict]):
        """
        Add resumes to FAISS index
        
        Args:
            resume_ids: List of resume identifiers
            embeddings: Numpy array of embeddings (n x dimension)
            metadatas: List of metadata dicts
        """
        if len(resume_ids) != len(embeddings) != len(metadatas):
            raise ValueError("All input lists must have the same length")
        
        # Train index if not trained and we have enough data
        if not hasattr(self, 'index_initialized') or not self.index_initialized:
            if embeddings.shape[0] >= 100:
                self._train_index(embeddings)
            else:
                # Use flat index for small datasets
                logger.info("Using flat index for small dataset")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index_initialized = True
        
        # Add to index
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.index.add(embeddings)
        
        # Store metadata
        self.resume_ids.extend(resume_ids)
        self.resume_metadata.extend(metadatas)
        
        # Save to disk
        self._persist()
        
        logger.info(f"Added {len(resume_ids)} resumes. Total: {len(self.resume_ids)}")
    
    def add_resume(self, resume_id: str, embedding: np.ndarray, metadata: Dict):
        """Add single resume to index"""
        self.add_resumes([resume_id], embedding.reshape(1, -1), [metadata])
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for similar resumes
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (ids, distances, metadatas)
        """
        if len(self.resume_ids) == 0:
            logger.warning("No resumes in index")
            return [], [], []
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, len(self.resume_ids)))
        
        # Get results
        result_ids = []
        result_distances = []
        result_metadatas = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.resume_ids):
                result_ids.append(self.resume_ids[idx])
                result_distances.append(float(distances[0][i]))
                result_metadatas.append(self.resume_metadata[idx])
        
        return result_ids, result_distances, result_metadatas
    
    def search_with_scores(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """
        Search and return similarity scores (converted from L2 distance)
        
        Returns:
            List of dicts with id, distance, similarity_score, metadata
        """
        ids, distances, metadatas = self.search_similar(query_embedding, k)
        
        results = []
        for idx, (resume_id, distance, metadata) in enumerate(zip(ids, distances, metadatas)):
            # Convert L2 distance to similarity score (0-1 range)
            # Using exponential decay: similarity = e^(-distance)
            similarity = np.exp(-distance)
            
            results.append({
                "id": resume_id,
                "distance": distance,
                "similarity_score": similarity,
                "match_percentage": f"{similarity * 100:.1f}%",
                "metadata": metadata,
                "rank": idx + 1
            })
        
        return results
    
    def _persist(self):
        """Save index and metadata to disk"""
        if self.index is not None:
            index_path = self.persist_path / "faiss.index"
            faiss.write_index(self.index, str(index_path))
            
            metadata_path = self.persist_path / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'ids': self.resume_ids,
                    'metadata': self.resume_metadata
                }, f)
            
            logger.info(f"Persisted index with {len(self.resume_ids)} resumes")
    
    def get_count(self) -> int:
        """Get total number of resumes in index"""
        return len(self.resume_ids)
    
    def get_all_ids(self) -> List[str]:
        """Get all resume IDs"""
        return self.resume_ids.copy()
    
    def clear(self):
        """Clear all data from index"""
        self._create_new_index()
        self.resume_ids = []
        self.resume_metadata = []
        self._persist()
        logger.info("Cleared all data from FAISS store")
    
    def remove_by_id(self, resume_id: str):
        """
        Remove a resume from the index (rebuilds index - expensive for large datasets)
        
        Note: FAISS doesn't support direct removal, so we rebuild
        """
        if resume_id not in self.resume_ids:
            logger.warning(f"Resume {resume_id} not found")
            return
        
        # Find index of resume
        idx = self.resume_ids.index(resume_id)
        
        # Remove from lists
        self.resume_ids.pop(idx)
        self.resume_metadata.pop(idx)
        
        # Rebuild index
        self._create_new_index()
        
        if len(self.resume_ids) > 0:
            # Need to re-add all embeddings (expensive)
            logger.warning("Rebuilding index after removal...")
            # You would need to store all embeddings to do this
            # For now, we just clear and require re-ingestion
            logger.info("Use clear() and re-ingest all resumes for complete removal")
        
        self._persist()


class FAISSIndexFactory:
    """Factory to create different types of FAISS indexes"""
    
    @staticmethod
    def create_flat_index(dimension: int):
        """Simple flat index (exact search, slower for large datasets)"""
        return faiss.IndexFlatL2(dimension)
    
    @staticmethod
    def create_ivf_index(dimension: int, nlist: int = 100):
        """IVF index (faster search, approximate)"""
        quantizer = faiss.IndexFlatL2(dimension)
        return faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    @staticmethod
    def create_hnsw_index(dimension: int, M: int = 32):
        """HNSW index (very fast search, good accuracy)"""
        return faiss.IndexHNSWFlat(dimension, M)