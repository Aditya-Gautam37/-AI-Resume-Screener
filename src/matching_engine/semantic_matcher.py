"""
Semantic Matcher Module
Matches resumes to job descriptions using cosine similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from loguru import logger

class SemanticMatcher:
    """Match resumes to job descriptions with semantic understanding"""
    
    def __init__(self, threshold: float = 0.65):
        """
        Initialize matcher
        
        Args:
            threshold: Minimum match score (0-1) to consider a match
        """
        self.threshold = threshold
        logger.info(f"SemanticMatcher initialized with threshold {threshold}")
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )
        return float(similarity[0][0])
    
    def match_resume_to_job(self, resume_embedding: np.ndarray, job_embedding: np.ndarray,
                           resume_entities: Dict, job_entities: Dict) -> Dict:
        """
        Calculate comprehensive match score between resume and job
        
        Args:
            resume_embedding: Resume embedding vector
            job_embedding: Job description embedding
            resume_entities: Extracted resume entities
            job_entities: Extracted job entities
            
        Returns:
            Dictionary with match scores
        """
        # Semantic similarity (70% weight)
        semantic_score = self.calculate_similarity(resume_embedding, job_embedding)
        
        # Entity matching score (30% weight)
        entity_score = self._calculate_entity_match(resume_entities, job_entities)
        
        # Weighted final score
        final_score = (semantic_score * 0.7) + (entity_score * 0.3)
        
        result = {
            "match_score": round(final_score, 3),
            "semantic_score": round(semantic_score, 3),
            "entity_score": round(entity_score, 3),
            "is_match": final_score >= self.threshold,
            "match_percentage": f"{final_score * 100:.1f}%"
        }
        
        return result
    
    def _calculate_entity_match(self, resume_entities: Dict, job_entities: Dict) -> float:
        """Calculate entity-based matching score"""
        total_score = 0.0
        weights = {
            "skills": 0.6,
            "education": 0.2,
            "experience": 0.2
        }
        
        # Skills match
        resume_skills = set([s.lower() for s in resume_entities.get("skills", [])])
        job_skills = set([s.lower() for s in job_entities.get("required_skills", [])])
        
        if job_skills:
            skill_match = len(resume_skills.intersection(job_skills)) / len(job_skills)
        else:
            skill_match = 0.5
        total_score += skill_match * weights["skills"]
        
        # Education match
        resume_edu = " ".join(resume_entities.get("education", [])).lower()
        required_edu = job_entities.get("required_education", "").lower()
        
        if required_edu and resume_edu:
            edu_match = 1.0 if required_edu in resume_edu else 0.5
        else:
            edu_match = 0.5
        total_score += edu_match * weights["education"]
        
        # Experience match
        resume_exp = resume_entities.get("years_experience", 0)
        required_exp = job_entities.get("required_experience", 0)
        
        if required_exp > 0:
            exp_match = min(1.0, resume_exp / required_exp)
        else:
            exp_match = 0.5
        total_score += exp_match * weights["experience"]
        
        return total_score
    
    def rank_candidates(self, matches: List[Dict]) -> List[Dict]:
        """Rank candidates by match score"""
        return sorted(matches, key=lambda x: x['match_score'], reverse=True)