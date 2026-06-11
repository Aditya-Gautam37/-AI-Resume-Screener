"""
Resume Ingestion Pipeline with FAISS
Fast vector search using Facebook AI Similarity Search
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_extraction.simple_parser import SimpleResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.faiss_store import FAISSVectorStore
from loguru import logger

# Configure logging
logger.add("logs/faiss_ingestion.log", rotation="500 MB")

def load_resume_file(file_path: Path) -> Dict:
    """Load and extract text from resume file"""
    try:
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.suffix == '.pdf':
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ' '.join([page.extract_text() for page in reader.pages])
        elif file_path.suffix == '.docx':
            import docx
            doc = docx.Document(file_path)
            text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return None
        
        return {
            "id": file_path.stem,
            "path": str(file_path),
            "text": text,
            "file_type": file_path.suffix
        }
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def main():
    """Main ingestion pipeline with FAISS"""
    print("=" * 60)
    print("AI RESUME SCREENER - FAISS INGESTION PIPELINE")
    print("=" * 60)
    
    # Initialize components
    print("\n[1/5] Initializing components...")
    parser = SimpleResumeParser()
    embedder = EmbeddingGenerator()
    dimension = embedder.get_embedding_dimension()
    vector_store = FAISSVectorStore(dimension=dimension)
    
    print(f"  ✓ Embedding dimension: {dimension}")
    print(f"  ✓ Existing resumes in FAISS: {vector_store.get_count()}")
    
    # Find all resume files
    resume_dir = Path("./data/raw/resumes")
    if not resume_dir.exists():
        resume_dir.mkdir(parents=True)
        print(f"\nCreated {resume_dir}")
        print("Please add resume files (.txt, .pdf, .docx) to this folder")
        return
    
    resume_files = [f for f in resume_dir.iterdir() if f.suffix in ['.txt', '.pdf', '.docx']]
    
    if not resume_files:
        print(f"\n❌ No resume files found in {resume_dir}")
        print("Please add .txt, .pdf, or .docx files and try again")
        return
    
    print(f"\n[2/5] Found {len(resume_files)} resume files")
    
    # Load resumes
    print("\n[3/5] Loading resume files...")
    resumes = []
    for file_path in resume_files:
        resume_data = load_resume_file(file_path)
        if resume_data:
            resumes.append(resume_data)
            print(f"  ✓ Loaded: {file_path.name}")
    
    if not resumes:
        print("❌ No valid resumes found")
        return
    
    # Extract entities
    print("\n[4/5] Extracting entities...")
    for resume in resumes:
        resume['entities'] = parser.extract_entities(resume['text'])
        print(f"  ✓ {resume['id']}: {len(resume['entities']['skills'])} skills, "
              f"{resume['entities']['years_experience']} years exp")
    
    # Generate embeddings
    print("\n[5/5] Generating embeddings and storing in FAISS...")
    texts = [r['text'] for r in resumes]
    embeddings = embedder.generate_embeddings(texts)
    
    # Prepare metadata
    resume_ids = []
    metadatas = []
    for i, resume in enumerate(resumes):
        resume_ids.append(resume['id'])
        metadatas.append({
            "id": resume['id'],
            "name": resume['entities'].get('name', 'Unknown'),
            "skills": resume['entities']['skills'],
            "education": resume['entities']['education'],
            "years_experience": resume['entities']['years_experience'],
            "file_path": resume['path']
        })
    
    # Add to FAISS
    vector_store.add_resumes(resume_ids, embeddings, metadatas)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ FAISS INGESTION COMPLETE!")
    print("=" * 60)
    print(f"\n📊 Statistics:")
    print(f"  - Total resumes processed: {len(resumes)}")
    print(f"  - Total resumes in FAISS: {vector_store.get_count()}")
    print(f"  - FAISS index type: IVF Flat")
    print(f"  - Embedding dimension: {dimension}")
    
    # Show sample
    print("\n📋 Sample Resume Data:")
    sample = resumes[0]
    print(f"  - ID: {sample['id']}")
    print(f"  - Name: {sample['entities'].get('name', 'Not found')}")
    print(f"  - Skills: {', '.join(sample['entities']['skills'][:5])}")
    print(f"  - Experience: {sample['entities']['years_experience']} years")
    
    print("\n✨ Ready for fast FAISS search! Run:")
    print("  python pipelines/match_job_faiss.py data/raw/job_descriptions/your_job.txt")

if __name__ == "__main__":
    main()