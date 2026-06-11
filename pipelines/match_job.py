"""
Job Matching Pipeline with FAISS
Fast similarity search using FAISS vector index
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_extraction.simple_parser import SimpleResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.faiss_store import FAISSVectorStore
from loguru import logger

logger.add("logs/faiss_matching.log", rotation="500 MB")

def extract_skills_from_jd(text: str) -> List[str]:
    """Automatically extract skills from job description"""
    text_lower = text.lower()
    
    # Comprehensive skill list
    skill_keywords = [
        # Programming Languages
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'swift',
        'kotlin', 'php', 'typescript', 'scala', 'perl', 'r', 'matlab', 'sql',
        
        # Web Frameworks
        'react', 'angular', 'vue', 'django', 'flask', 'node.js', 'express', 'spring',
        'asp.net', 'rails', 'laravel', 'next.js', 'nuxt', 'svelte',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins',
        'gitlab', 'github actions', 'terraform', 'ansible', 'puppet', 'chef',
        'cloudformation', 'lambda', 'ec2', 's3', 'rds',
        
        # Databases
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'cassandra',
        'dynamodb', 'oracle', 'mssql', 'sqlite', 'firebase',
        
        # Data Science & ML
        'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras',
        'matplotlib', 'seaborn', 'jupyter', 'spark', 'hadoop', 'airflow',
        'tableau', 'power bi', 'llm', 'openai', 'langchain',
        
        # Mobile
        'ios', 'android', 'react native', 'flutter', 'swiftui', 'kotlin',
        
        # Other
        'git', 'linux', 'agile', 'scrum', 'jira', 'confluence', 'rest api',
        'graphql', 'microservices', 'ci/cd', 'devops', 'mlops', 'data analysis'
    ]
    
    extracted_skills = []
    for skill in skill_keywords:
        if skill in text_lower:
            extracted_skills.append(skill)
    
    # Also look for skills mentioned in bullet points or with capital letters
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower()
        # Look for lines with bullet points or skill indicators
        if any(indicator in line_lower for indicator in ['•', '-', '*', 'skills:', 'requirements:', 'qualifications:']):
            words = re.findall(r'\b[a-z][a-z\s-]+[a-z]\b', line_lower)
            for word in words:
                if len(word) > 2 and word not in extracted_skills and len(word) < 30:
                    # Check if it might be a skill
                    if any(tech in word for tech in ['python', 'java', 'aws', 'sql', 'react']):
                        extracted_skills.append(word)
    
    return list(set(extracted_skills))  # Remove duplicates

def load_job_description(file_path: str) -> Dict:
    """Load job description from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Auto-extract skills
        skills = extract_skills_from_jd(text)
        
        # Extract experience requirement
        exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', text.lower())
        required_exp = int(exp_match.group(1)) if exp_match else 0
        
        return {
            "path": file_path,
            "text": text,
            "name": Path(file_path).stem,
            "required_skills": skills,
            "required_experience": required_exp
        }
    except Exception as e:
        logger.error(f"Error loading job description: {e}")
        return None

def display_results(results: List[Dict], job_skills: List[str], required_exp: int):
    """Display FAISS search results in formatted table"""
    print("\n" + "=" * 100)
    print(f"{'Rank':<6} {'Resume ID':<25} {'Match Score':<12} {'Skills Match':<15} {'Experience':<12}")
    print("=" * 100)
    
    for result in results[:10]:
        score = result['similarity_score']
        metadata = result['metadata']
        
        # Calculate skills match percentage
        if job_skills:
            matched_skills = len(set(metadata.get('skills', [])) & set(job_skills))
            skills_match = f"{matched_skills}/{len(job_skills)} ({matched_skills/len(job_skills)*100:.0f}%)"
        else:
            skills_match = "N/A"
        
        # Experience match indicator
        exp_years = metadata.get('years_experience', 0)
        exp_status = "✅" if exp_years >= required_exp else "⚠️" if exp_years >= required_exp/2 else "❌"
        
        # Determine match strength
        if score >= 0.7:
            status = "✅ Strong"
        elif score >= 0.5:
            status = "⚠️ Potential"
        else:
            status = "❌ Weak"
        
        print(f"{result['rank']:<6} {result['id'][:24]:<25} {score*100:>6.1f}%      "
              f"{skills_match:<20} {exp_status} {exp_years}yrs")
        
        # Show top matching skills if match is good
        if score >= 0.6 and job_skills and metadata.get('skills'):
            matched = set(metadata.get('skills', [])) & set(job_skills)
            if matched:
                print(f"       📚 Matched Skills: {', '.join(list(matched)[:5])}")
    
    print("=" * 100)

def main():
    """Main matching pipeline with FAISS"""
    print("=" * 60)
    print("AI RESUME SCREENER - FAISS MATCHING ENGINE")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("\n❌ Usage: python pipelines/match_job_faiss.py <job_description_file>")
        print("Example: python pipelines/match_job_faiss.py data/raw/job_descriptions/jd.txt")
        sys.exit(1)
    
    job_file = sys.argv[1]
    
    if not os.path.exists(job_file):
        print(f"\n❌ Job description file not found: {job_file}")
        sys.exit(1)
    
    # Load job description (auto-extracts skills)
    print(f"\n[1/4] Loading job description...")
    job_data = load_job_description(job_file)
    if not job_data:
        print("❌ Failed to load job description")
        sys.exit(1)
    
    print(f"  ✓ Loaded: {Path(job_file).name}")
    print(f"  ✓ Length: {len(job_data['text'])} characters")
    print(f"  ✓ Auto-extracted skills: {len(job_data['required_skills'])} skills found")
    if job_data['required_skills']:
        print(f"    Skills: {', '.join(job_data['required_skills'][:10])}")
    print(f"  ✓ Required experience: {job_data['required_experience']} years")
    
    # Initialize components
    print("\n[2/4] Initializing components...")
    parser = SimpleResumeParser()
    embedder = EmbeddingGenerator()
    dimension = embedder.get_embedding_dimension()
    vector_store = FAISSVectorStore(dimension=dimension)
    
    # Check if FAISS has resumes
    resume_count = vector_store.get_count()
    if resume_count == 0:
        print("\n❌ No resumes found in FAISS database!")
        print("Please run ingestion first: python pipelines/ingest_resumes_faiss.py")
        sys.exit(1)
    
    print(f"  ✓ Found {resume_count} resumes in FAISS")
    print(f"  ✓ FAISS index type: IVF Flat")
    
    # Extract job entities (for additional context)
    print("\n[3/4] Analyzing job requirements...")
    job_entities = parser.extract_entities(job_data['text'])
    
    # Use auto-extracted skills (no user input needed)
    job_skills = job_data['required_skills']
    required_exp = job_data['required_experience']
    
    if not job_skills:
        print("  ⚠️ No skills auto-detected. Using semantic matching only.")
    
    print(f"  ✓ Skills for matching: {len(job_skills)}")
    print(f"  ✓ Experience requirement: {required_exp} years")
    
    # Generate job embedding
    print("\n[4/4] Searching FAISS index...")
    job_embedding = embedder.generate_single_embedding(job_data['text'])
    
    # Search with FAISS (super fast!)
    import time
    start_time = time.time()
    results = vector_store.search_with_scores(job_embedding, k=min(20, resume_count))
    search_time = time.time() - start_time
    
    print(f"  ✓ FAISS search completed in {search_time*1000:.2f}ms")
    
    # Filter and rank results
    filtered_results = []
    for result in results:
        metadata = result['metadata']
        
        # Apply experience filter
        if required_exp > 0 and metadata.get('years_experience', 0) < required_exp:
            # Still include but with penalty
            exp_penalty = 0.8
            result['similarity_score'] *= exp_penalty
            result['exp_filtered'] = True
        else:
            result['exp_filtered'] = False
        
        # Calculate skills match and boost score
        if job_skills:
            matched_skills = len(set(metadata.get('skills', [])) & set(job_skills))
            skills_match_pct = matched_skills / len(job_skills)
            
            # Boost score based on skills match (up to 30% boost)
            boost = 1 + (skills_match_pct * 0.3)
            result['similarity_score'] = min(1.0, result['similarity_score'] * boost)
            result['skills_match_count'] = matched_skills
            result['skills_match_pct'] = skills_match_pct
        
        filtered_results.append(result)
    
    # Sort by final score
    filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Display results
    display_results(filtered_results, job_skills, required_exp)
    
    # Performance stats
    print(f"\n⚡ Performance:")
    print(f"  - Search time: {search_time*1000:.2f}ms")
    print(f"  - Resumes searched: {resume_count}")
    if search_time > 0:
        print(f"  - Speed: {resume_count/search_time:.0f} resumes/second")
    else:
        print(f"  - Speed: >100,000 resumes/second (instantaneous)")
    
    # Show top match details
    if filtered_results:
        print("\n🏆 Top Match Details:")
        top = filtered_results[0]
        metadata = top['metadata']
        print(f"  - Resume: {top['id']}")
        print(f"  - Match Score: {top['similarity_score']*100:.1f}%")
        print(f"  - Skills Found: {', '.join(metadata.get('skills', [])[:8])}")
        
        if job_skills:
            matched = set(metadata.get('skills', [])) & set(job_skills)
            print(f"  - Matched Requirements: {len(matched)}/{len(job_skills)} skills")
            if matched:
                print(f"    ✓ Matched: {', '.join(list(matched)[:5])}")
        
        print(f"  - Experience: {metadata.get('years_experience', 0)} years")
        print(f"  - Education: {metadata.get('education', ['N/A'])[0] if metadata.get('education') else 'N/A'}")
    
    print("\n✨ Matching complete!")

if __name__ == "__main__":
    main()