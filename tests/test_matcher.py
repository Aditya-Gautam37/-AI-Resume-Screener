"""
Unit tests for semantic matcher
"""

import pytest
import numpy as np
from src.matching_engine.semantic_matcher import SemanticMatcher

class TestSemanticMatcher:
    
    def setup_method(self):
        self.matcher = SemanticMatcher(threshold=0.65)
    
    def test_similarity_calculation(self):
        """Test cosine similarity calculation"""
        emb1 = np.array([1, 0, 0])
        emb2 = np.array([1, 0, 0])
        
        similarity = self.matcher.calculate_similarity(emb1, emb2)
        assert similarity == 1.0
        
        emb3 = np.array([0, 1, 0])
        similarity = self.matcher.calculate_similarity(emb1, emb3)
        assert similarity == 0.0
    
    def test_entity_match_score(self):
        """Test entity matching score"""
        resume_entities = {
            "skills": ["python", "sql", "java", "aws"],
            "education": ["MIT"],
            "years_experience": 5
        }
        
        job_entities = {
            "required_skills": ["python", "sql", "aws"],
            "required_education": "MIT",
            "required_experience": 3
        }
        
        score = self.matcher._calculate_entity_match(resume_entities, job_entities)
        assert score > 0.8
        
    def test_match_result(self):
        """Test complete match result"""
        resume_emb = np.random.rand(384)
        job_emb = np.random.rand(384)
        
        resume_entities = {"skills": ["python"], "education": [], "years_experience": 2}
        job_entities = {"required_skills": ["python"], "required_education": "", "required_experience": 0}
        
        result = self.matcher.match_resume_to_job(
            resume_emb, job_emb, resume_entities, job_entities
        )
        
        assert "match_score" in result
        assert "is_match" in result
        assert 0 <= result["match_score"] <= 1    
    def test_ranking(self):
        """Test candidate ranking"""
        matches = [
            {"match_score": 0.9, "resume_id": "resume1"},
            {"match_score": 0.5, "resume_id": "resume2"},
            {"match_score": 0.7, "resume_id": "resume3"}
        ]
        
        ranked = self.matcher.rank_candidates(matches)
        assert ranked[0]["match_score"] == 0.9
        assert ranked[1]["match_score"] == 0.7
        assert ranked[2]["match_score"] == 0.5