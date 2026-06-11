import re
from typing import Dict, List, Optional

class SimpleResumeParser:
    def __init__(self):
        self.skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 
                      'kubernetes', 'react', 'angular', 'django', 'flask', 
                      'tensorflow', 'pytorch', 'pandas', 'numpy', 'mongodb', 
                      'postgresql', 'mysql', 'git', 'linux', 'devops', 'ci/cd']
    
    def extract_entities(self, text: str) -> Dict:
        text_lower = text.lower()
        return {
            "name": self._extract_name(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "skills": [s for s in self.skills if s in text_lower],
            "education": self._extract_education(text),
            "years_experience": self._extract_experience(text),
            "job_titles": [],
            "companies": []
        }
    
    def _extract_name(self, text: str):
        lines = text.strip().split('\n')
        return lines[0].strip() if lines else None
    
    def _extract_email(self, text: str):
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str):
        match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        return match.group(0) if match else None
    
    def _extract_education(self, text: str):
        keywords = ['bachelor', 'master', 'phd', 'bs', 'ms', 'degree', 'university', 'college']
        return [line.strip() for line in text.split('\n') 
                if any(k in line.lower() for k in keywords)][:3]
    
    def _extract_experience(self, text: str):
        match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*experience', text.lower())
        return int(match.group(1)) if match else 0