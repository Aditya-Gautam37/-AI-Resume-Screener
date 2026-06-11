"""
Resume Parser Module
Extracts entities from resumes using spaCy
"""

import spacy
import re
from typing import Dict, List, Optional
from loguru import logger

class ResumeParser:
    """Extract key information from resumes"""
    
    def __init__(self):
        logger.info("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("Model not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
            self.nlp = spacy.load("en_core_web_lg")
        
        # Common skills dictionary
        self.skills_dict = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'sql'],
            'web': ['react', 'angular', 'vue', 'django', 'flask', 'node.js', 'html', 'css'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform'],
            'data': ['tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'tableau', 'power bi'],
            'database': ['mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch']
        }
        
        self.all_skills = [skill.lower() for category in self.skills_dict.values() for skill in category]
        logger.info("ResumeParser initialized")
    
    def extract_entities(self, text: str) -> Dict:
        """
        Extract entities from resume text
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary of extracted entities
        """
        # Process with spaCy (limit text length for performance)
        doc = self.nlp(text[:500000])
        
        entities = {
            "name": None,
            "email": None,
            "phone": None,
            "skills": [],
            "education": [],
            "job_titles": [],
            "companies": [],
            "years_experience": 0
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not entities["name"]:
                entities["name"] = ent.text
            elif ent.label_ == "ORG":
                if any(word in ent.text.lower() for word in ['university', 'college', 'institute']):
                    entities["education"].append(ent.text)
                else:
                    entities["companies"].append(ent.text)
        
        # Extract skills using keyword matching
        text_lower = text.lower()
        for skill in self.all_skills:
            if skill in text_lower:
                entities["skills"].append(skill)
        
        # Remove duplicates
        entities["skills"] = list(set(entities["skills"]))
        entities["education"] = list(set(entities["education"]))
        entities["companies"] = list(set(entities["companies"]))
        
        # Extract contact information
        entities["email"] = self._extract_email(text)
        entities["phone"] = self._extract_phone(text)
        
        # Estimate experience years
        entities["years_experience"] = self._extract_experience(text)
        
        logger.debug(f"Extracted {len(entities['skills'])} skills from resume")
        return entities
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address using regex"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number using regex"""
        patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',
            r'\b\d{10}\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _extract_experience(self, text: str) -> int:
        """Extract years of experience"""
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        # If no explicit years, estimate from employment dates
        date_pattern = r'(19|20)\d{2}'
        dates = re.findall(date_pattern, text)
        if dates:
            years = max([int(d) for d in dates])
            current_year = 2024
            return max(0, current_year - years)
        
        return 0