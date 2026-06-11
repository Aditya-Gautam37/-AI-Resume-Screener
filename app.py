

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os
import sys
import time
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.entity_extraction.simple_parser import SimpleResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.faiss_store import FAISSVectorStore

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp { background: #111318; color: #d4d8e0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #16181f !important;
        border-right: 1px solid #252830;
    }
    [data-testid="stSidebar"] * { color: #b0b6c2 !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 { color: #d4d8e0 !important; font-weight: 600; }

    /* ── Buttons — all styles ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.1rem !important;
        transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
        cursor: pointer !important;
        letter-spacing: 0.1px !important;
    }
    /* Primary */
    .stButton > button[kind="primary"] {
        background: #2c5282 !important;
        color: #e8edf5 !important;
        border: 1px solid #2c5282 !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2a4a75 !important;
        border-color: #2a4a75 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:active {
        background: #243f63 !important;
    }
    /* Secondary / default */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind]) {
        background: #1e2028 !important;
        color: #9aa0ad !important;
        border: 1px solid #2e3140 !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind]):hover {
        background: #252830 !important;
        color: #d4d8e0 !important;
        border-color: #3c4153 !important;
    }

    /* ── Header ── */
    .main-header {
        font-size: 1.9rem;
        font-weight: 700;
        color: #d4d8e0;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.3px;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #5c6270;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.2px;
    }

    /* ── Score badges ── */
    .match-score-high {
        background: #1a2e1a;
        border: 1px solid #2d5a2d;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #6abf6a;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }
    .match-score-medium {
        background: #2a2410;
        border: 1px solid #5a4a10;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #c9a84c;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }
    .match-score-low {
        background: #2a1414;
        border: 1px solid #5a2020;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #c06060;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }

    /* ── Skill badges ── */
    .skill-badge {
        background: #1e2028;
        border: 1px solid #2e3140;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        color: #8a9ab5;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.1rem;
    }

    /* ── Weight cards ── */
    .weight-card {
        background: #16181f;
        border: 1px solid #252830;
        border-top: 2px solid #2c5282;
        padding: 0.8rem 0.5rem;
        border-radius: 8px;
        text-align: center;
        color: #d4d8e0;
    }

    /* ── Component cards ── */
    .component-card {
        padding: 0.9rem 0.6rem;
        border-radius: 8px;
        text-align: center;
        background: #16181f;
        border: 1px solid #252830;
        color: #d4d8e0;
    }
    .contribution-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
        margin-top: 0.35rem;
        background: #1e2028;
        border: 1px solid #2e3140;
        color: #8a9ab5;
    }

    /* ── Candidate card ── */
    .candidate-card {
        background: #16181f;
        border: 1px solid #252830;
        border-radius: 10px;
        padding: 1.4rem 1.6rem 1rem;
        margin-bottom: 1.2rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4a5060;
        margin-bottom: 0.5rem;
    }

    /* ── Rank badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #1e2028;
        border: 1px solid #2e3140;
        color: #8a9ab5;
        font-weight: 700;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        flex-shrink: 0;
    }
    .rank-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #d4d8e0;
    }

    /* ── Progress bars ── */
    .stProgress > div > div { background: #1e2028 !important; border-radius: 3px; }
    .stProgress > div > div > div > div { background: #2c5282 !important; border-radius: 3px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #16181f;
        border-radius: 8px;
        padding: 0.25rem;
        border: 1px solid #252830;
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        border-radius: 6px;
        color: #5c6270;
        padding: 0.35rem 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2028 !important;
        color: #d4d8e0 !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #16181f;
        border: 1px solid #252830 !important;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary { color: #5c6270; font-size: 0.875rem; }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] { color: #8a9ab5 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #4a5060 !important; }

    /* ── Alerts ── */
    .stAlert { border-radius: 6px !important; }

    /* ── Dividers ── */
    hr { border-color: #1e2028 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #16181f !important;
        border: 1px dashed #252830 !important;
        border-radius: 8px !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] .rc-slider-rail { background: #252830; }
    [data-testid="stSlider"] .rc-slider-track { background: #2c5282; }
    [data-testid="stSlider"] .rc-slider-handle { border-color: #2c5282; background: #2c5282; }

    /* ── Tables ── */
    table { color: #d4d8e0 !important; }
    thead tr th { color: #5c6270 !important; border-bottom: 1px solid #252830 !important; }
    tbody tr td { border-color: #1e2028 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = {}
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = {}
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'parser' not in st.session_state:
    st.session_state.parser = None

FIXED_WEIGHTS = {
    'skills': 0.35,
    'location': 0.10,
    'experience': 0.10,
    'qualification': 0.25,
    'job_title': 0.20
}

@st.cache_resource
def init_models():
    with st.spinner("Loading AI models..."):
        parser = SimpleResumeParser()
        embedder = EmbeddingGenerator()
        dimension = embedder.get_embedding_dimension()
        vector_store = FAISSVectorStore(dimension=dimension, persist_path="./data/embeddings/streamlit_faiss")
        return parser, embedder, vector_store

def extract_skills_from_text(text: str) -> List[str]:
    skill_keywords = [
        'python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
        'react', 'angular', 'vue', 'django', 'flask', 'tensorflow', 'pytorch',
        'pandas', 'numpy', 'mongodb', 'postgresql', 'mysql', 'git', 'linux',
        'devops', 'ci/cd', 'agile', 'scrum', 'machine learning', 'deep learning',
        'nlp', 'data analysis', 'tableau', 'power bi', 'excel', 'c++', 'c#',
        'ruby', 'php', 'html', 'css', 'node.js', 'express', 'spring', 'azure',
        'gcp', 'jenkins', 'terraform', 'ansible', 'redis', 'elasticsearch',
        'spark', 'hadoop', 'airflow', 'kafka', 'rabbitmq', 'graphql', 'rest api'
    ]
    text_lower = text.lower()
    return list(set(s for s in skill_keywords if s in text_lower))

def extract_location(text: str) -> str:
    location_keywords = ['remote', 'onsite', 'hybrid', 'new york', 'san francisco',
                         'london', 'bangalore', 'mumbai', 'delhi', 'singapore',
                         'austin', 'seattle', 'boston', 'chicago', 'los angeles']
    text_lower = text.lower()
    for kw in location_keywords:
        if kw in text_lower:
            return kw
    m = re.search(r'(?:location|based in|office)[:\s]+([A-Za-z\s,]+)', text_lower)
    return m.group(1).strip() if m else "not specified"

def extract_qualification(text: str) -> Tuple[str, int]:
    qualifications = {
        'phd': (5, 'PhD'), 'doctorate': (5, 'PhD'), 'doctoral': (5, 'PhD'),
        'master': (4, "Master's"), 'masters': (4, "Master's"), 'ms': (4, "Master's"),
        'm.sc': (4, "Master's"), 'mba': (4, 'MBA'),
        'bachelor': (3, "Bachelor's"), 'bachelors': (3, "Bachelor's"), 'bs': (3, "Bachelor's"),
        'b.sc': (3, "Bachelor's"), 'ba': (3, "Bachelor's"),
        'associate': (2, "Associate's"), 'diploma': (2, 'Diploma'),
        'high school': (1, 'High School'), 'secondary': (1, 'High School')
    }
    text_lower = text.lower()
    highest_level, highest_degree = 0, "Not Specified"
    for degree, (level, formal_name) in qualifications.items():
        if degree in text_lower and level > highest_level:
            highest_level, highest_degree = level, formal_name
    return highest_degree, highest_level

def extract_job_titles(text: str) -> List[str]:
    title_keywords = [
        'software engineer', 'data scientist', 'devops engineer', 'frontend developer',
        'backend developer', 'full stack developer', 'product manager', 'project manager',
        'business analyst', 'data analyst', 'machine learning engineer', 'cloud architect',
        'system administrator', 'database administrator', 'security engineer', 'qa engineer',
        'tech lead', 'engineering manager', 'solutions architect', 'technical writer'
    ]
    text_lower = text.lower()
    found = [t for t in title_keywords if t in text_lower]
    matches = re.findall(r'(?:title|position|role|seeking|hiring for)[:\s]+([A-Za-z\s/]+)', text_lower)
    found.extend([m.strip() for m in matches[:2]])
    return list(set(found)) if found else ['not specified']

def calculate_title_similarity(resume_title: str, jd_title: str) -> float:
    if not resume_title or resume_title == 'not specified':
        return 0.5
    if not jd_title or jd_title == 'not specified':
        return 0.5
    rt, jt = resume_title.lower(), jd_title.lower()
    if rt == jt:
        return 1.0
    similarity = SequenceMatcher(None, rt, jt).ratio()
    synonyms = {
        'engineer': ['engineer', 'developer', 'programmer', 'coder', 'software engineer'],
        'developer': ['developer', 'engineer', 'programmer', 'software engineer'],
        'scientist': ['scientist', 'analyst', 'researcher', 'data scientist'],
        'manager': ['manager', 'lead', 'director', 'head', 'team lead'],
        'analyst': ['analyst', 'associate', 'coordinator', 'business analyst'],
        'architect': ['architect', 'designer', 'planner', 'solutions architect']
    }
    for key, syns in synonyms.items():
        if key in rt and any(s in jt for s in syns):
            return 0.85
        if key in jt and any(s in rt for s in syns):
            return 0.85
    return max(similarity, 0.3)

def calculate_location_match(resume_location: str, jd_location: str) -> float:
    if not resume_location or resume_location == 'not specified':
        return 0.5
    if not jd_location or jd_location == 'not specified':
        return 0.5
    rl, jl = resume_location.lower(), jd_location.lower()
    if rl == jl: return 1.0
    if 'remote' in jl: return 1.0
    if 'remote' in rl: return 0.9
    if rl in jl or jl in rl: return 0.8
    return 0.4

def calculate_qualification_match(resume_education: List[str], jd_qual_level: int) -> float:
    qual_levels = {
        'phd': 5, 'doctorate': 5, 'doctoral': 5,
        'master': 4, 'masters': 4, 'ms': 4, 'm.sc': 4, 'mba': 4,
        'bachelor': 3, 'bachelors': 3, 'bs': 3, 'b.sc': 3, 'ba': 3,
        'associate': 2, 'diploma': 2, 'high school': 1, 'secondary': 1
    }
    if not resume_education:
        return 0.5
    resume_level = 0
    for edu in resume_education:
        edu_lower = edu.lower()
        for qual, level in qual_levels.items():
            if qual in edu_lower and level > resume_level:
                resume_level = level
    if resume_level == 0:
        return 0.5
    if jd_qual_level > 0:
        return min(1.0, resume_level / jd_qual_level)
    return 0.7 if resume_level >= 3 else 0.5

def process_resumes(uploaded_files) -> int:
    parser = st.session_state.parser
    with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
        for file in uploaded_files:
            try:
                text = file.getvalue().decode('utf-8')
                entities = parser.extract_entities(text)
                all_skills = list(set(entities.get('skills', []) + extract_skills_from_text(text)))
                qualification, qual_level = extract_qualification(text)
                st.session_state.resume_data[file.name] = {
                    'name': entities.get('name', file.name.replace('.txt', '').replace('_', ' ').title()),
                    'email': entities.get('email'),
                    'phone': entities.get('phone'),
                    'skills': all_skills,
                    'education': entities.get('education', []),
                    'experience': entities.get('years_experience', 0),
                    'location': extract_location(text),
                    'qualification': qualification,
                    'qualification_level': qual_level,
                    'job_titles': extract_job_titles(text),
                    'text': text,
                    'original_file': file.name
                }
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
    return len(st.session_state.resume_data)

def process_jd(uploaded_files) -> int:
    for file in uploaded_files:
        try:
            text = file.getvalue().decode('utf-8')
            qualification, qual_level = extract_qualification(text)
            exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', text.lower())
            st.session_state.jd_data[file.name] = {
                'name': file.name.replace('.txt', '').replace('_', ' ').title(),
                'text': text,
                'required_skills': extract_skills_from_text(text),
                'required_experience': int(exp_match.group(1)) if exp_match else 0,
                'location': extract_location(text),
                'required_qualification': qualification,
                'required_qualification_level': qual_level if qual_level > 0 else 3,
                'job_titles': extract_job_titles(text),
                'full_text': text
            }
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return len(st.session_state.jd_data)

def generate_embeddings() -> bool:
    if not st.session_state.resume_data:
        st.warning("No resumes to process")
        return False
    embedder = st.session_state.embedder
    vector_store = st.session_state.vector_store
    with st.spinner(f"Generating embeddings for {len(st.session_state.resume_data)} resumes..."):
        for resume_name, resume_info in st.session_state.resume_data.items():
            embedding = embedder.generate_single_embedding(resume_info['text'])
            vector_store.add_resume(
                resume_id=resume_name,
                embedding=embedding,
                metadata={
                    'name': resume_info['name'],
                    'skills': resume_info['skills'],
                    'experience': resume_info['experience'],
                    'education': resume_info['education'],
                    'location': resume_info['location'],
                    'qualification': resume_info['qualification'],
                    'qualification_level': resume_info['qualification_level'],
                    'job_titles': resume_info['job_titles']
                }
            )
    st.session_state.embeddings_generated = True
    return True

def calculate_weighted_match(jd_info, resume_info, resume_metadata, semantic_score, weights):
    scores = {}
    jd_skills = set(jd_info['required_skills'])
    resume_skills = set(resume_metadata.get('skills', []))
    scores['skills'] = len(jd_skills & resume_skills) / len(jd_skills) if jd_skills else semantic_score
    scores['location'] = calculate_location_match(
        resume_info.get('location', 'not specified'), jd_info.get('location', 'not specified'))
    req_exp = jd_info['required_experience']
    res_exp = resume_metadata.get('experience', 0)
    scores['experience'] = min(1.0, res_exp / req_exp) if req_exp > 0 else (0.7 if res_exp > 0 else 0.5)
    scores['qualification'] = calculate_qualification_match(
        resume_metadata.get('education', []), jd_info.get('required_qualification_level', 3))
    jd_titles = jd_info.get('job_titles', ['not specified'])
    res_titles = resume_metadata.get('job_titles', ['not specified'])
    if jd_titles and res_titles and jd_titles[0] != 'not specified':
        title_matches = [calculate_title_similarity(rt, jd_titles[0]) for rt in res_titles]
        scores['job_title'] = max(title_matches) if title_matches else 0.5
    else:
        scores['job_title'] = 0.5
    total = sum(scores[k] * weights[k] for k in weights)
    return total, scores

def match_resume_to_jd(jd_name, top_k=5, weights=None):
    if not st.session_state.embeddings_generated:
        return []
    if weights is None:
        weights = FIXED_WEIGHTS
    jd_info = st.session_state.jd_data.get(jd_name)
    if not jd_info:
        return []
    embedder = st.session_state.embedder
    vector_store = st.session_state.vector_store
    jd_embedding = embedder.generate_single_embedding(jd_info['text'])
    results = vector_store.search_with_scores(jd_embedding, k=min(top_k * 2, len(st.session_state.resume_data)))
    matches = []
    for result in results:
        metadata = result['metadata']
        resume_id = result['id']
        resume_info = st.session_state.resume_data.get(resume_id, {})
        final_score, component_scores = calculate_weighted_match(
            jd_info, resume_info, metadata, result['similarity_score'], weights)
        contributions = {k: component_scores[k] * weights[k] * 100 for k in weights}
        matches.append({
            'resume_id': resume_id,
            'resume_name': resume_info.get('name', resume_id),
            'final_score': final_score,
            'semantic_score': result['similarity_score'],
            'scores': component_scores,
            'contributions': contributions,
            'matched_skills': list(set(jd_info['required_skills']) & set(metadata.get('skills', [])))[:10],
            'resume_skills': metadata.get('skills', [])[:15],
            'resume_experience': metadata.get('experience', 0),
            'resume_education': metadata.get('education', []),
            'resume_location': resume_info.get('location', 'not specified'),
            'resume_job_titles': metadata.get('job_titles', []),
            'email': resume_info.get('email'),
            'phone': resume_info.get('phone')
        })
    matches.sort(key=lambda x: x['final_score'], reverse=True)
    return matches[:top_k]

def get_score_color(score: float) -> str:
    if score >= 0.7: return "match-score-high"
    elif score >= 0.5: return "match-score-medium"
    return "match-score-low"

def get_score_emoji(score: float) -> str:
    if score >= 0.7: return "↑"
    elif score >= 0.5: return "→"
    return "↓"

def display_matches(matches, min_score, jd_info, weights):
    if not matches:
        st.info("No matches found.")
        return

    matches = [m for m in matches if m['final_score'] >= min_score]
    if not matches:
        st.warning(f"No candidates above {min_score*100:.0f}% match score.")
        return

    st.success(f"{len(matches)} candidate{'s' if len(matches) != 1 else ''} matched")

    components = [
        ('skills',        '🎯 Skills',        ),
        ('location',      '📍 Location',      ),
        ('experience',    '📅 Experience',    ),
        ('qualification', '🎓 Qualification', ),
        ('job_title',     '💼 Job Title',     ),
    ]

    for i, match in enumerate(matches, 1):
        score = match['final_score']
        score_class = get_score_color(score)
        direction = get_score_emoji(score)

        st.markdown('<div class="candidate-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(
                f'<span class="rank-badge">{i}</span>'
                f'<span class="rank-name">{match["resume_name"]}</span>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f'<div class="{score_class}">{direction} {score*100:.1f}%</div>', unsafe_allow_html=True)
            st.caption("Overall match")
        with col3:
            st.markdown(
                f'<div style="background:#1e2028;border:1px solid #2e3140;border-radius:6px;'
                f'padding:0.4rem 0.8rem;text-align:center;">'
                f'<div style="color:#4a5060;font-size:0.7rem;margin-bottom:2px;">SEMANTIC</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#8a9ab5;">'
                f'{match["semantic_score"]*100:.0f}%</div></div>',
                unsafe_allow_html=True
            )

        st.divider()

        # Component breakdown
        st.markdown('<div class="section-label">Score breakdown</div>', unsafe_allow_html=True)
        for comp_key, comp_name in components:
            comp_score = match['scores'][comp_key] * 100
            weight = weights[comp_key] * 100
            contribution = match['contributions'][comp_key]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<span style="font-size:0.82rem;font-weight:500;color:#b0b6c2;">{comp_name}</span>'
                    f'<span style="font-size:0.72rem;color:#4a5060;margin-left:0.4rem;">({weight:.0f}% weight)</span>',
                    unsafe_allow_html=True
                )
                st.progress(comp_score / 100, text=f"{comp_score:.0f}%")
            with col_b:
                st.markdown(
                    f'<div style="text-align:right;padding-top:1.3rem;">'
                    f'<span class="contribution-badge">+{contribution:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # Summary row
        st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for idx, (comp_key, comp_name) in enumerate(components):
            with cols[idx]:
                comp_score = match['scores'][comp_key] * 100
                weight = weights[comp_key] * 100
                contribution = match['contributions'][comp_key]
                st.markdown(f"""
                <div class="component-card">
                    <div style="font-size:0.72rem;color:#4a5060;margin-bottom:0.25rem;">{comp_name}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#8a9ab5;">{comp_score:.0f}%</div>
                    <div style="font-size:0.68rem;color:#3a4050;margin-top:0.1rem;">wt {weight:.0f}%</div>
                    <div class="contribution-badge">+{contribution:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Skills
        if match['matched_skills']:
            st.markdown('<div class="section-label">Matched skills</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                badges = " ".join([f'<span class="skill-badge">{s}</span>' for s in match['matched_skills'][:8]])
                st.markdown(badges, unsafe_allow_html=True)
            with col2:
                skills_score = match['scores']['skills'] * 100
                skills_contribution = match['contributions']['skills']
                st.markdown(
                    f'<div style="background:#1e2028;border:1px solid #2e3140;border-radius:6px;'
                    f'padding:0.6rem;text-align:center;margin-top:0.1rem;">'
                    f'<div style="font-size:0.7rem;color:#4a5060;">Skills match</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#8a9ab5;">{skills_score:.0f}%</div>'
                    f'<div style="font-size:0.7rem;color:#4a5060;">+{skills_contribution:.1f}% to total</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Requirements
        st.markdown('<div class="section-label" style="margin-top:0.8rem;">Requirements comparison</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            exp_score = match['scores']['experience'] * 100
            exp_contribution = match['contributions']['experience']
            st.markdown(f"""
            **📅 Experience**
            - Required: {jd_info.get('required_experience', 0)} years
            - Candidate: {match['resume_experience']} years
            - Match: {exp_score:.0f}% → Adds {exp_contribution:.1f}%
            """)
            loc_score = match['scores']['location'] * 100
            loc_contribution = match['contributions']['location']
            st.markdown(f"""
            **📍 Location**
            - Required: {jd_info.get('location', 'Not specified').title()}
            - Candidate: {match['resume_location'].title()}
            - Match: {loc_score:.0f}% → Adds {loc_contribution:.1f}%
            """)
        with col2:
            qual_score = match['scores']['qualification'] * 100
            qual_contribution = match['contributions']['qualification']
            st.markdown(f"""
            **🎓 Qualification**
            - Required: {jd_info.get('required_qualification', 'Bachelor')}
            - Candidate: {', '.join(match['resume_education'][:2]) if match['resume_education'] else 'Not specified'}
            - Match: {qual_score:.0f}% → Adds {qual_contribution:.1f}%
            """)
            title_score = match['scores']['job_title'] * 100
            title_contribution = match['contributions']['job_title']
            st.markdown(f"""
            **💼 Job Title**
            - Required: {jd_info.get('job_titles', ['Not specified'])[0].title()}
            - Candidate: {match['resume_job_titles'][0].title() if match['resume_job_titles'] else 'Not specified'}
            - Match: {title_score:.0f}% → Adds {title_contribution:.1f}%
            """)

        with st.expander("📐 Score calculation details"):
            st.markdown("**Total Score = Σ (Component Score × Weight)**")
            st.markdown("| Component | Score | Weight | Contribution |")
            st.markdown("|-----------|-------|--------|--------------|")
            for comp_key, comp_name in components:
                comp_score = match['scores'][comp_key] * 100
                weight = weights[comp_key] * 100
                contribution = match['contributions'][comp_key]
                st.markdown(f"| {comp_name} | {comp_score:.0f}% | {weight:.0f}% | **+{contribution:.1f}%** |")
            st.markdown("---")
            st.markdown("*Example: Skills 86% × 35% = +30.1% contribution*")

        with st.expander("📄 Full candidate profile"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {match['resume_name']}")
                st.markdown(f"**Email:** {match.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {match.get('phone', 'N/A')}")
                st.markdown(f"**Location:** {match['resume_location'].title()}")
            with col2:
                st.markdown(f"**Experience:** {match['resume_experience']} years")
                st.markdown(f"**Education:** {', '.join(match['resume_education'][:3]) if match['resume_education'] else 'N/A'}")
                st.markdown(f"**Job Titles:** {', '.join(match.get('resume_job_titles', [])[:3])}")
            st.markdown("**All Skills:**")
            skill_cols = st.columns(4)
            for idx, skill in enumerate(match['resume_skills'][:20]):
                with skill_cols[idx % 4]:
                    st.markdown(f"- {skill}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")


def main():
    st.markdown('<h1 class="main-header">AI Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Weighted matching — Skills 35% · Qualification 25% · Job Title 20% · Location 10% · Experience 10%</p>', unsafe_allow_html=True)

    if st.session_state.parser is None:
        parser, embedder, vector_store = init_models()
        st.session_state.parser = parser
        st.session_state.embedder = embedder
        st.session_state.vector_store = vector_store

    with st.sidebar:
        st.markdown("#### Upload Documents")
        st.markdown("**Resumes** (up to 10)")
        resume_files = st.file_uploader(
            "Upload .txt files", type=['txt'], accept_multiple_files=True, key="resume_uploader")
        if resume_files:
            if st.button("Process Resumes", type="primary", use_container_width=True):
                count = process_resumes(resume_files)
                st.success(f"Processed {count} resumes")
                st.session_state.embeddings_generated = False
                st.rerun()

        st.divider()

        st.markdown("**Job Descriptions** (up to 2)")
        jd_files = st.file_uploader(
            "Upload .txt files", type=['txt'], accept_multiple_files=True, key="jd_uploader")
        if jd_files:
            if st.button("Process JDs", type="primary", use_container_width=True):
                count = process_jd(jd_files)
                st.success(f"Processed {count} job descriptions")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resumes", len(st.session_state.resume_data))
        with col2:
            st.metric("JDs", len(st.session_state.jd_data))

        if len(st.session_state.resume_data) > 0 and not st.session_state.embeddings_generated:
            if st.button("Generate Embeddings", type="primary", use_container_width=True):
                if generate_embeddings():
                    st.success("Embeddings ready")
                    st.rerun()
        elif st.session_state.embeddings_generated:
            st.success("✓ Embeddings ready")

        st.divider()

        if st.button("Clear All Data", use_container_width=True):
            st.session_state.resume_data = {}
            st.session_state.jd_data = {}
            st.session_state.embeddings_generated = False
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
            st.success("Cleared")
            st.rerun()

    if not st.session_state.jd_data:
        st.info("Upload job descriptions in the sidebar to get started.")
    elif not st.session_state.resume_data:
        st.info("Upload resumes in the sidebar to match against the job descriptions.")
    elif not st.session_state.embeddings_generated:
        st.warning("Click **Generate Embeddings** in the sidebar to enable matching.")
    else:
        weights = FIXED_WEIGHTS

        st.markdown('<div class="section-label">Matching weights (fixed)</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        weight_items = [
            (weights['skills'] * 100,        '🎯 Skills'),
            (weights['location'] * 100,      '📍 Location'),
            (weights['experience'] * 100,    '📅 Experience'),
            (weights['qualification'] * 100, '🎓 Qualification'),
            (weights['job_title'] * 100,     '💼 Job Title'),
        ]
        for col, (w, name) in zip(cols, weight_items):
            with col:
                st.markdown(
                    f'<div class="weight-card">'
                    f'<div style="font-size:0.75rem;color:#4a5060;margin-bottom:0.25rem;">{name}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:#8a9ab5;">{w:.0f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        jd_list = list(st.session_state.jd_data.keys())
        if len(jd_list) == 1:
            selected_jd = jd_list[0]
            jd_info = st.session_state.jd_data[selected_jd]
            with st.expander(f"Job description: {selected_jd}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Required Skills ({len(jd_info['required_skills'])}):**")
                    for skill in jd_info['required_skills'][:10]:
                        st.markdown(f"- {skill}")
                with col2:
                    st.markdown(f"**Experience:** {jd_info['required_experience']} years")
                    st.markdown(f"**Location:** {jd_info['location'].title()}")
                with col3:
                    st.markdown(f"**Qualification:** {jd_info['required_qualification']}")
                    st.markdown(f"**Target Role:** {jd_info['job_titles'][0].title() if jd_info['job_titles'] else 'Not specified'}")

            col1, col2 = st.columns([2, 1])
            with col1:
                top_k = st.slider("Number of candidates:", 1, 10, 5)
            with col2:
                min_score = st.slider("Minimum match score:", 0.0, 1.0, 0.3, 0.05)

            if st.button("Find Matching Candidates", type="primary", use_container_width=True):
                with st.spinner("Calculating matches..."):
                    matches = match_resume_to_jd(selected_jd, top_k=top_k, weights=weights)
                    display_matches(matches, min_score, jd_info, weights)
        else:
            jd_tabs = st.tabs([f"{jd.replace('.txt', '')[:25]}" for jd in jd_list])
            for idx, (jd_name, tab) in enumerate(zip(jd_list, jd_tabs)):
                with tab:
                    jd_info = st.session_state.jd_data[jd_name]
                    with st.expander("Job description details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Required Skills ({len(jd_info['required_skills'])}):**")
                            for skill in jd_info['required_skills'][:8]:
                                st.markdown(f"- {skill}")
                        with col2:
                            st.markdown(f"**Experience:** {jd_info['required_experience']} years")
                            st.markdown(f"**Location:** {jd_info['location'].title()}")
                        with col3:
                            st.markdown(f"**Qualification:** {jd_info['required_qualification']}")
                            st.markdown(f"**Target Role:** {jd_info['job_titles'][0].title() if jd_info['job_titles'] else 'Not specified'}")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        top_k = st.slider("Candidates", 1, 10, 5, key=f"topk_{idx}")
                    with col2:
                        min_score = st.slider("Min score", 0.0, 1.0, 0.3, 0.05, key=f"minscore_{idx}")

                    if st.button(f"Find Matches — {jd_name.replace('.txt', '')}", key=f"match_btn_{idx}", type="primary"):
                        with st.spinner("Calculating matches..."):
                            matches = match_resume_to_jd(jd_name, top_k=top_k, weights=weights)
                            display_matches(matches, min_score, jd_info, weights)


if __name__ == "__main__":
"""
AI Resume Screener - Complete Streamlit UI with Fixed Weights
Skills 35% | Location 10% | Experience 10% | Qualification 25% | Job Title 20%
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import os
import sys
import time
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.entity_extraction.simple_parser import SimpleResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.faiss_store import FAISSVectorStore

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp { background: #111318; color: #d4d8e0; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #16181f !important;
        border-right: 1px solid #252830;
    }
    [data-testid="stSidebar"] * { color: #b0b6c2 !important; }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 { color: #d4d8e0 !important; font-weight: 600; }

    /* ── Buttons — all styles ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
        padding: 0.45rem 1.1rem !important;
        transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease !important;
        cursor: pointer !important;
        letter-spacing: 0.1px !important;
    }
    /* Primary */
    .stButton > button[kind="primary"] {
        background: #2c5282 !important;
        color: #e8edf5 !important;
        border: 1px solid #2c5282 !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2a4a75 !important;
        border-color: #2a4a75 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:active {
        background: #243f63 !important;
    }
    /* Secondary / default */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind]) {
        background: #1e2028 !important;
        color: #9aa0ad !important;
        border: 1px solid #2e3140 !important;
    }
    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind]):hover {
        background: #252830 !important;
        color: #d4d8e0 !important;
        border-color: #3c4153 !important;
    }

    /* ── Header ── */
    .main-header {
        font-size: 1.9rem;
        font-weight: 700;
        color: #d4d8e0;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.3px;
    }
    .sub-header {
        font-size: 0.85rem;
        color: #5c6270;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: 0.2px;
    }

    /* ── Score badges ── */
    .match-score-high {
        background: #1a2e1a;
        border: 1px solid #2d5a2d;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #6abf6a;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }
    .match-score-medium {
        background: #2a2410;
        border: 1px solid #5a4a10;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #c9a84c;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }
    .match-score-low {
        background: #2a1414;
        border: 1px solid #5a2020;
        padding: 0.45rem 1rem;
        border-radius: 6px;
        color: #c06060;
        font-weight: 700;
        text-align: center;
        font-size: 1.5rem;
    }

    /* ── Skill badges ── */
    .skill-badge {
        background: #1e2028;
        border: 1px solid #2e3140;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        color: #8a9ab5;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.1rem;
    }

    /* ── Weight cards ── */
    .weight-card {
        background: #16181f;
        border: 1px solid #252830;
        border-top: 2px solid #2c5282;
        padding: 0.8rem 0.5rem;
        border-radius: 8px;
        text-align: center;
        color: #d4d8e0;
    }

    /* ── Component cards ── */
    .component-card {
        padding: 0.9rem 0.6rem;
        border-radius: 8px;
        text-align: center;
        background: #16181f;
        border: 1px solid #252830;
        color: #d4d8e0;
    }
    .contribution-badge {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
        margin-top: 0.35rem;
        background: #1e2028;
        border: 1px solid #2e3140;
        color: #8a9ab5;
    }

    /* ── Candidate card ── */
    .candidate-card {
        background: #16181f;
        border: 1px solid #252830;
        border-radius: 10px;
        padding: 1.4rem 1.6rem 1rem;
        margin-bottom: 1.2rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4a5060;
        margin-bottom: 0.5rem;
    }

    /* ── Rank badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #1e2028;
        border: 1px solid #2e3140;
        color: #8a9ab5;
        font-weight: 700;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        flex-shrink: 0;
    }
    .rank-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #d4d8e0;
    }

    /* ── Progress bars ── */
    .stProgress > div > div { background: #1e2028 !important; border-radius: 3px; }
    .stProgress > div > div > div > div { background: #2c5282 !important; border-radius: 3px; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #16181f;
        border-radius: 8px;
        padding: 0.25rem;
        border: 1px solid #252830;
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        border-radius: 6px;
        color: #5c6270;
        padding: 0.35rem 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: #1e2028 !important;
        color: #d4d8e0 !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #16181f;
        border: 1px solid #252830 !important;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary { color: #5c6270; font-size: 0.875rem; }

    /* ── Metrics ── */
    [data-testid="stMetricValue"] { color: #8a9ab5 !important; font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { color: #4a5060 !important; }

    /* ── Alerts ── */
    .stAlert { border-radius: 6px !important; }

    /* ── Dividers ── */
    hr { border-color: #1e2028 !important; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #16181f !important;
        border: 1px dashed #252830 !important;
        border-radius: 8px !important;
    }

    /* ── Sliders ── */
    [data-testid="stSlider"] .rc-slider-rail { background: #252830; }
    [data-testid="stSlider"] .rc-slider-track { background: #2c5282; }
    [data-testid="stSlider"] .rc-slider-handle { border-color: #2c5282; background: #2c5282; }

    /* ── Tables ── */
    table { color: #d4d8e0 !important; }
    thead tr th { color: #5c6270 !important; border-bottom: 1px solid #252830 !important; }
    tbody tr td { border-color: #1e2028 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = {}
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = {}
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'parser' not in st.session_state:
    st.session_state.parser = None

FIXED_WEIGHTS = {
    'skills': 0.35,
    'location': 0.10,
    'experience': 0.10,
    'qualification': 0.25,
    'job_title': 0.20
}

@st.cache_resource
def init_models():
    with st.spinner("Loading AI models..."):
        parser = SimpleResumeParser()
        embedder = EmbeddingGenerator()
        dimension = embedder.get_embedding_dimension()
        vector_store = FAISSVectorStore(dimension=dimension, persist_path="./data/embeddings/streamlit_faiss")
        return parser, embedder, vector_store

def extract_skills_from_text(text: str) -> List[str]:
    skill_keywords = [
        'python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
        'react', 'angular', 'vue', 'django', 'flask', 'tensorflow', 'pytorch',
        'pandas', 'numpy', 'mongodb', 'postgresql', 'mysql', 'git', 'linux',
        'devops', 'ci/cd', 'agile', 'scrum', 'machine learning', 'deep learning',
        'nlp', 'data analysis', 'tableau', 'power bi', 'excel', 'c++', 'c#',
        'ruby', 'php', 'html', 'css', 'node.js', 'express', 'spring', 'azure',
        'gcp', 'jenkins', 'terraform', 'ansible', 'redis', 'elasticsearch',
        'spark', 'hadoop', 'airflow', 'kafka', 'rabbitmq', 'graphql', 'rest api'
    ]
    text_lower = text.lower()
    return list(set(s for s in skill_keywords if s in text_lower))

def extract_location(text: str) -> str:
    location_keywords = ['remote', 'onsite', 'hybrid', 'new york', 'san francisco',
                         'london', 'bangalore', 'mumbai', 'delhi', 'singapore',
                         'austin', 'seattle', 'boston', 'chicago', 'los angeles']
    text_lower = text.lower()
    for kw in location_keywords:
        if kw in text_lower:
            return kw
    m = re.search(r'(?:location|based in|office)[:\s]+([A-Za-z\s,]+)', text_lower)
    return m.group(1).strip() if m else "not specified"

def extract_qualification(text: str) -> Tuple[str, int]:
    qualifications = {
        'phd': (5, 'PhD'), 'doctorate': (5, 'PhD'), 'doctoral': (5, 'PhD'),
        'master': (4, "Master's"), 'masters': (4, "Master's"), 'ms': (4, "Master's"),
        'm.sc': (4, "Master's"), 'mba': (4, 'MBA'),
        'bachelor': (3, "Bachelor's"), 'bachelors': (3, "Bachelor's"), 'bs': (3, "Bachelor's"),
        'b.sc': (3, "Bachelor's"), 'ba': (3, "Bachelor's"),
        'associate': (2, "Associate's"), 'diploma': (2, 'Diploma'),
        'high school': (1, 'High School'), 'secondary': (1, 'High School')
    }
    text_lower = text.lower()
    highest_level, highest_degree = 0, "Not Specified"
    for degree, (level, formal_name) in qualifications.items():
        if degree in text_lower and level > highest_level:
            highest_level, highest_degree = level, formal_name
    return highest_degree, highest_level

def extract_job_titles(text: str) -> List[str]:
    title_keywords = [
        'software engineer', 'data scientist', 'devops engineer', 'frontend developer',
        'backend developer', 'full stack developer', 'product manager', 'project manager',
        'business analyst', 'data analyst', 'machine learning engineer', 'cloud architect',
        'system administrator', 'database administrator', 'security engineer', 'qa engineer',
        'tech lead', 'engineering manager', 'solutions architect', 'technical writer'
    ]
    text_lower = text.lower()
    found = [t for t in title_keywords if t in text_lower]
    matches = re.findall(r'(?:title|position|role|seeking|hiring for)[:\s]+([A-Za-z\s/]+)', text_lower)
    found.extend([m.strip() for m in matches[:2]])
    return list(set(found)) if found else ['not specified']

def calculate_title_similarity(resume_title: str, jd_title: str) -> float:
    if not resume_title or resume_title == 'not specified':
        return 0.5
    if not jd_title or jd_title == 'not specified':
        return 0.5
    rt, jt = resume_title.lower(), jd_title.lower()
    if rt == jt:
        return 1.0
    similarity = SequenceMatcher(None, rt, jt).ratio()
    synonyms = {
        'engineer': ['engineer', 'developer', 'programmer', 'coder', 'software engineer'],
        'developer': ['developer', 'engineer', 'programmer', 'software engineer'],
        'scientist': ['scientist', 'analyst', 'researcher', 'data scientist'],
        'manager': ['manager', 'lead', 'director', 'head', 'team lead'],
        'analyst': ['analyst', 'associate', 'coordinator', 'business analyst'],
        'architect': ['architect', 'designer', 'planner', 'solutions architect']
    }
    for key, syns in synonyms.items():
        if key in rt and any(s in jt for s in syns):
            return 0.85
        if key in jt and any(s in rt for s in syns):
            return 0.85
    return max(similarity, 0.3)

def calculate_location_match(resume_location: str, jd_location: str) -> float:
    if not resume_location or resume_location == 'not specified':
        return 0.5
    if not jd_location or jd_location == 'not specified':
        return 0.5
    rl, jl = resume_location.lower(), jd_location.lower()
    if rl == jl: return 1.0
    if 'remote' in jl: return 1.0
    if 'remote' in rl: return 0.9
    if rl in jl or jl in rl: return 0.8
    return 0.4

def calculate_qualification_match(resume_education: List[str], jd_qual_level: int) -> float:
    qual_levels = {
        'phd': 5, 'doctorate': 5, 'doctoral': 5,
        'master': 4, 'masters': 4, 'ms': 4, 'm.sc': 4, 'mba': 4,
        'bachelor': 3, 'bachelors': 3, 'bs': 3, 'b.sc': 3, 'ba': 3,
        'associate': 2, 'diploma': 2, 'high school': 1, 'secondary': 1
    }
    if not resume_education:
        return 0.5
    resume_level = 0
    for edu in resume_education:
        edu_lower = edu.lower()
        for qual, level in qual_levels.items():
            if qual in edu_lower and level > resume_level:
                resume_level = level
    if resume_level == 0:
        return 0.5
    if jd_qual_level > 0:
        return min(1.0, resume_level / jd_qual_level)
    return 0.7 if resume_level >= 3 else 0.5

def process_resumes(uploaded_files) -> int:
    parser = st.session_state.parser
    with st.spinner(f"Processing {len(uploaded_files)} resumes..."):
        for file in uploaded_files:
            try:
                text = file.getvalue().decode('utf-8')
                entities = parser.extract_entities(text)
                all_skills = list(set(entities.get('skills', []) + extract_skills_from_text(text)))
                qualification, qual_level = extract_qualification(text)
                st.session_state.resume_data[file.name] = {
                    'name': entities.get('name', file.name.replace('.txt', '').replace('_', ' ').title()),
                    'email': entities.get('email'),
                    'phone': entities.get('phone'),
                    'skills': all_skills,
                    'education': entities.get('education', []),
                    'experience': entities.get('years_experience', 0),
                    'location': extract_location(text),
                    'qualification': qualification,
                    'qualification_level': qual_level,
                    'job_titles': extract_job_titles(text),
                    'text': text,
                    'original_file': file.name
                }
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
    return len(st.session_state.resume_data)

def process_jd(uploaded_files) -> int:
    for file in uploaded_files:
        try:
            text = file.getvalue().decode('utf-8')
            qualification, qual_level = extract_qualification(text)
            exp_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience', text.lower())
            st.session_state.jd_data[file.name] = {
                'name': file.name.replace('.txt', '').replace('_', ' ').title(),
                'text': text,
                'required_skills': extract_skills_from_text(text),
                'required_experience': int(exp_match.group(1)) if exp_match else 0,
                'location': extract_location(text),
                'required_qualification': qualification,
                'required_qualification_level': qual_level if qual_level > 0 else 3,
                'job_titles': extract_job_titles(text),
                'full_text': text
            }
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return len(st.session_state.jd_data)

def generate_embeddings() -> bool:
    if not st.session_state.resume_data:
        st.warning("No resumes to process")
        return False
    embedder = st.session_state.embedder
    vector_store = st.session_state.vector_store
    with st.spinner(f"Generating embeddings for {len(st.session_state.resume_data)} resumes..."):
        for resume_name, resume_info in st.session_state.resume_data.items():
            embedding = embedder.generate_single_embedding(resume_info['text'])
            vector_store.add_resume(
                resume_id=resume_name,
                embedding=embedding,
                metadata={
                    'name': resume_info['name'],
                    'skills': resume_info['skills'],
                    'experience': resume_info['experience'],
                    'education': resume_info['education'],
                    'location': resume_info['location'],
                    'qualification': resume_info['qualification'],
                    'qualification_level': resume_info['qualification_level'],
                    'job_titles': resume_info['job_titles']
                }
            )
    st.session_state.embeddings_generated = True
    return True

def calculate_weighted_match(jd_info, resume_info, resume_metadata, semantic_score, weights):
    scores = {}
    jd_skills = set(jd_info['required_skills'])
    resume_skills = set(resume_metadata.get('skills', []))
    scores['skills'] = len(jd_skills & resume_skills) / len(jd_skills) if jd_skills else semantic_score
    scores['location'] = calculate_location_match(
        resume_info.get('location', 'not specified'), jd_info.get('location', 'not specified'))
    req_exp = jd_info['required_experience']
    res_exp = resume_metadata.get('experience', 0)
    scores['experience'] = min(1.0, res_exp / req_exp) if req_exp > 0 else (0.7 if res_exp > 0 else 0.5)
    scores['qualification'] = calculate_qualification_match(
        resume_metadata.get('education', []), jd_info.get('required_qualification_level', 3))
    jd_titles = jd_info.get('job_titles', ['not specified'])
    res_titles = resume_metadata.get('job_titles', ['not specified'])
    if jd_titles and res_titles and jd_titles[0] != 'not specified':
        title_matches = [calculate_title_similarity(rt, jd_titles[0]) for rt in res_titles]
        scores['job_title'] = max(title_matches) if title_matches else 0.5
    else:
        scores['job_title'] = 0.5
    total = sum(scores[k] * weights[k] for k in weights)
    return total, scores

def match_resume_to_jd(jd_name, top_k=5, weights=None):
    if not st.session_state.embeddings_generated:
        return []
    if weights is None:
        weights = FIXED_WEIGHTS
    jd_info = st.session_state.jd_data.get(jd_name)
    if not jd_info:
        return []
    embedder = st.session_state.embedder
    vector_store = st.session_state.vector_store
    jd_embedding = embedder.generate_single_embedding(jd_info['text'])
    results = vector_store.search_with_scores(jd_embedding, k=min(top_k * 2, len(st.session_state.resume_data)))
    matches = []
    for result in results:
        metadata = result['metadata']
        resume_id = result['id']
        resume_info = st.session_state.resume_data.get(resume_id, {})
        final_score, component_scores = calculate_weighted_match(
            jd_info, resume_info, metadata, result['similarity_score'], weights)
        contributions = {k: component_scores[k] * weights[k] * 100 for k in weights}
        matches.append({
            'resume_id': resume_id,
            'resume_name': resume_info.get('name', resume_id),
            'final_score': final_score,
            'semantic_score': result['similarity_score'],
            'scores': component_scores,
            'contributions': contributions,
            'matched_skills': list(set(jd_info['required_skills']) & set(metadata.get('skills', [])))[:10],
            'resume_skills': metadata.get('skills', [])[:15],
            'resume_experience': metadata.get('experience', 0),
            'resume_education': metadata.get('education', []),
            'resume_location': resume_info.get('location', 'not specified'),
            'resume_job_titles': metadata.get('job_titles', []),
            'email': resume_info.get('email'),
            'phone': resume_info.get('phone')
        })
    matches.sort(key=lambda x: x['final_score'], reverse=True)
    return matches[:top_k]

def get_score_color(score: float) -> str:
    if score >= 0.7: return "match-score-high"
    elif score >= 0.5: return "match-score-medium"
    return "match-score-low"

def get_score_emoji(score: float) -> str:
    if score >= 0.7: return "↑"
    elif score >= 0.5: return "→"
    return "↓"

def display_matches(matches, min_score, jd_info, weights):
    if not matches:
        st.info("No matches found.")
        return

    matches = [m for m in matches if m['final_score'] >= min_score]
    if not matches:
        st.warning(f"No candidates above {min_score*100:.0f}% match score.")
        return

    st.success(f"{len(matches)} candidate{'s' if len(matches) != 1 else ''} matched")

    components = [
        ('skills',        '🎯 Skills',        ),
        ('location',      '📍 Location',      ),
        ('experience',    '📅 Experience',    ),
        ('qualification', '🎓 Qualification', ),
        ('job_title',     '💼 Job Title',     ),
    ]

    for i, match in enumerate(matches, 1):
        score = match['final_score']
        score_class = get_score_color(score)
        direction = get_score_emoji(score)

        st.markdown('<div class="candidate-card">', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(
                f'<span class="rank-badge">{i}</span>'
                f'<span class="rank-name">{match["resume_name"]}</span>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f'<div class="{score_class}">{direction} {score*100:.1f}%</div>', unsafe_allow_html=True)
            st.caption("Overall match")
        with col3:
            st.markdown(
                f'<div style="background:#1e2028;border:1px solid #2e3140;border-radius:6px;'
                f'padding:0.4rem 0.8rem;text-align:center;">'
                f'<div style="color:#4a5060;font-size:0.7rem;margin-bottom:2px;">SEMANTIC</div>'
                f'<div style="font-size:1.3rem;font-weight:700;color:#8a9ab5;">'
                f'{match["semantic_score"]*100:.0f}%</div></div>',
                unsafe_allow_html=True
            )

        st.divider()

        # Component breakdown
        st.markdown('<div class="section-label">Score breakdown</div>', unsafe_allow_html=True)
        for comp_key, comp_name in components:
            comp_score = match['scores'][comp_key] * 100
            weight = weights[comp_key] * 100
            contribution = match['contributions'][comp_key]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(
                    f'<span style="font-size:0.82rem;font-weight:500;color:#b0b6c2;">{comp_name}</span>'
                    f'<span style="font-size:0.72rem;color:#4a5060;margin-left:0.4rem;">({weight:.0f}% weight)</span>',
                    unsafe_allow_html=True
                )
                st.progress(comp_score / 100, text=f"{comp_score:.0f}%")
            with col_b:
                st.markdown(
                    f'<div style="text-align:right;padding-top:1.3rem;">'
                    f'<span class="contribution-badge">+{contribution:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

        st.divider()

        # Summary row
        st.markdown('<div class="section-label">At a glance</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for idx, (comp_key, comp_name) in enumerate(components):
            with cols[idx]:
                comp_score = match['scores'][comp_key] * 100
                weight = weights[comp_key] * 100
                contribution = match['contributions'][comp_key]
                st.markdown(f"""
                <div class="component-card">
                    <div style="font-size:0.72rem;color:#4a5060;margin-bottom:0.25rem;">{comp_name}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:#8a9ab5;">{comp_score:.0f}%</div>
                    <div style="font-size:0.68rem;color:#3a4050;margin-top:0.1rem;">wt {weight:.0f}%</div>
                    <div class="contribution-badge">+{contribution:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Skills
        if match['matched_skills']:
            st.markdown('<div class="section-label">Matched skills</div>', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                badges = " ".join([f'<span class="skill-badge">{s}</span>' for s in match['matched_skills'][:8]])
                st.markdown(badges, unsafe_allow_html=True)
            with col2:
                skills_score = match['scores']['skills'] * 100
                skills_contribution = match['contributions']['skills']
                st.markdown(
                    f'<div style="background:#1e2028;border:1px solid #2e3140;border-radius:6px;'
                    f'padding:0.6rem;text-align:center;margin-top:0.1rem;">'
                    f'<div style="font-size:0.7rem;color:#4a5060;">Skills match</div>'
                    f'<div style="font-size:1.1rem;font-weight:700;color:#8a9ab5;">{skills_score:.0f}%</div>'
                    f'<div style="font-size:0.7rem;color:#4a5060;">+{skills_contribution:.1f}% to total</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Requirements
        st.markdown('<div class="section-label" style="margin-top:0.8rem;">Requirements comparison</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            exp_score = match['scores']['experience'] * 100
            exp_contribution = match['contributions']['experience']
            st.markdown(f"""
            **📅 Experience**
            - Required: {jd_info.get('required_experience', 0)} years
            - Candidate: {match['resume_experience']} years
            - Match: {exp_score:.0f}% → Adds {exp_contribution:.1f}%
            """)
            loc_score = match['scores']['location'] * 100
            loc_contribution = match['contributions']['location']
            st.markdown(f"""
            **📍 Location**
            - Required: {jd_info.get('location', 'Not specified').title()}
            - Candidate: {match['resume_location'].title()}
            - Match: {loc_score:.0f}% → Adds {loc_contribution:.1f}%
            """)
        with col2:
            qual_score = match['scores']['qualification'] * 100
            qual_contribution = match['contributions']['qualification']
            st.markdown(f"""
            **🎓 Qualification**
            - Required: {jd_info.get('required_qualification', 'Bachelor')}
            - Candidate: {', '.join(match['resume_education'][:2]) if match['resume_education'] else 'Not specified'}
            - Match: {qual_score:.0f}% → Adds {qual_contribution:.1f}%
            """)
            title_score = match['scores']['job_title'] * 100
            title_contribution = match['contributions']['job_title']
            st.markdown(f"""
            **💼 Job Title**
            - Required: {jd_info.get('job_titles', ['Not specified'])[0].title()}
            - Candidate: {match['resume_job_titles'][0].title() if match['resume_job_titles'] else 'Not specified'}
            - Match: {title_score:.0f}% → Adds {title_contribution:.1f}%
            """)

        with st.expander("📐 Score calculation details"):
            st.markdown("**Total Score = Σ (Component Score × Weight)**")
            st.markdown("| Component | Score | Weight | Contribution |")
            st.markdown("|-----------|-------|--------|--------------|")
            for comp_key, comp_name in components:
                comp_score = match['scores'][comp_key] * 100
                weight = weights[comp_key] * 100
                contribution = match['contributions'][comp_key]
                st.markdown(f"| {comp_name} | {comp_score:.0f}% | {weight:.0f}% | **+{contribution:.1f}%** |")
            st.markdown("---")
            st.markdown("*Example: Skills 86% × 35% = +30.1% contribution*")

        with st.expander("📄 Full candidate profile"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {match['resume_name']}")
                st.markdown(f"**Email:** {match.get('email', 'N/A')}")
                st.markdown(f"**Phone:** {match.get('phone', 'N/A')}")
                st.markdown(f"**Location:** {match['resume_location'].title()}")
            with col2:
                st.markdown(f"**Experience:** {match['resume_experience']} years")
                st.markdown(f"**Education:** {', '.join(match['resume_education'][:3]) if match['resume_education'] else 'N/A'}")
                st.markdown(f"**Job Titles:** {', '.join(match.get('resume_job_titles', [])[:3])}")
            st.markdown("**All Skills:**")
            skill_cols = st.columns(4)
            for idx, skill in enumerate(match['resume_skills'][:20]):
                with skill_cols[idx % 4]:
                    st.markdown(f"- {skill}")

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")


def main():
    st.markdown('<h1 class="main-header">AI Resume Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Weighted matching — Skills 35% · Qualification 25% · Job Title 20% · Location 10% · Experience 10%</p>', unsafe_allow_html=True)

    if st.session_state.parser is None:
        parser, embedder, vector_store = init_models()
        st.session_state.parser = parser
        st.session_state.embedder = embedder
        st.session_state.vector_store = vector_store

    with st.sidebar:
        st.markdown("#### Upload Documents")
        st.markdown("**Resumes** (up to 10)")
        resume_files = st.file_uploader(
            "Upload .txt files", type=['txt'], accept_multiple_files=True, key="resume_uploader")
        if resume_files:
            if st.button("Process Resumes", type="primary", use_container_width=True):
                count = process_resumes(resume_files)
                st.success(f"Processed {count} resumes")
                st.session_state.embeddings_generated = False
                st.rerun()

        st.divider()

        st.markdown("**Job Descriptions** (up to 2)")
        jd_files = st.file_uploader(
            "Upload .txt files", type=['txt'], accept_multiple_files=True, key="jd_uploader")
        if jd_files:
            if st.button("Process JDs", type="primary", use_container_width=True):
                count = process_jd(jd_files)
                st.success(f"Processed {count} job descriptions")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Resumes", len(st.session_state.resume_data))
        with col2:
            st.metric("JDs", len(st.session_state.jd_data))

        if len(st.session_state.resume_data) > 0 and not st.session_state.embeddings_generated:
            if st.button("Generate Embeddings", type="primary", use_container_width=True):
                if generate_embeddings():
                    st.success("Embeddings ready")
                    st.rerun()
        elif st.session_state.embeddings_generated:
            st.success("✓ Embeddings ready")

        st.divider()

        if st.button("Clear All Data", use_container_width=True):
            st.session_state.resume_data = {}
            st.session_state.jd_data = {}
            st.session_state.embeddings_generated = False
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
            st.success("Cleared")
            st.rerun()

    if not st.session_state.jd_data:
        st.info("Upload job descriptions in the sidebar to get started.")
    elif not st.session_state.resume_data:
        st.info("Upload resumes in the sidebar to match against the job descriptions.")
    elif not st.session_state.embeddings_generated:
        st.warning("Click **Generate Embeddings** in the sidebar to enable matching.")
    else:
        weights = FIXED_WEIGHTS

        st.markdown('<div class="section-label">Matching weights (fixed)</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        weight_items = [
            (weights['skills'] * 100,        '🎯 Skills'),
            (weights['location'] * 100,      '📍 Location'),
            (weights['experience'] * 100,    '📅 Experience'),
            (weights['qualification'] * 100, '🎓 Qualification'),
            (weights['job_title'] * 100,     '💼 Job Title'),
        ]
        for col, (w, name) in zip(cols, weight_items):
            with col:
                st.markdown(
                    f'<div class="weight-card">'
                    f'<div style="font-size:0.75rem;color:#4a5060;margin-bottom:0.25rem;">{name}</div>'
                    f'<div style="font-size:1.4rem;font-weight:700;color:#8a9ab5;">{w:.0f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        jd_list = list(st.session_state.jd_data.keys())
        if len(jd_list) == 1:
            selected_jd = jd_list[0]
            jd_info = st.session_state.jd_data[selected_jd]
            with st.expander(f"Job description: {selected_jd}", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Required Skills ({len(jd_info['required_skills'])}):**")
                    for skill in jd_info['required_skills'][:10]:
                        st.markdown(f"- {skill}")
                with col2:
                    st.markdown(f"**Experience:** {jd_info['required_experience']} years")
                    st.markdown(f"**Location:** {jd_info['location'].title()}")
                with col3:
                    st.markdown(f"**Qualification:** {jd_info['required_qualification']}")
                    st.markdown(f"**Target Role:** {jd_info['job_titles'][0].title() if jd_info['job_titles'] else 'Not specified'}")

            col1, col2 = st.columns([2, 1])
            with col1:
                top_k = st.slider("Number of candidates:", 1, 10, 5)
            with col2:
                min_score = st.slider("Minimum match score:", 0.0, 1.0, 0.3, 0.05)

            if st.button("Find Matching Candidates", type="primary", use_container_width=True):
                with st.spinner("Calculating matches..."):
                    matches = match_resume_to_jd(selected_jd, top_k=top_k, weights=weights)
                    display_matches(matches, min_score, jd_info, weights)
        else:
            jd_tabs = st.tabs([f"{jd.replace('.txt', '')[:25]}" for jd in jd_list])
            for idx, (jd_name, tab) in enumerate(zip(jd_list, jd_tabs)):
                with tab:
                    jd_info = st.session_state.jd_data[jd_name]
                    with st.expander("Job description details", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**Required Skills ({len(jd_info['required_skills'])}):**")
                            for skill in jd_info['required_skills'][:8]:
                                st.markdown(f"- {skill}")
                        with col2:
                            st.markdown(f"**Experience:** {jd_info['required_experience']} years")
                            st.markdown(f"**Location:** {jd_info['location'].title()}")
                        with col3:
                            st.markdown(f"**Qualification:** {jd_info['required_qualification']}")
                            st.markdown(f"**Target Role:** {jd_info['job_titles'][0].title() if jd_info['job_titles'] else 'Not specified'}")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        top_k = st.slider("Candidates", 1, 10, 5, key=f"topk_{idx}")
                    with col2:
                        min_score = st.slider("Min score", 0.0, 1.0, 0.3, 0.05, key=f"minscore_{idx}")

                    if st.button(f"Find Matches — {jd_name.replace('.txt', '')}", key=f"match_btn_{idx}", type="primary"):
                        with st.spinner("Calculating matches..."):
                            matches = match_resume_to_jd(jd_name, top_k=top_k, weights=weights)
                            display_matches(matches, min_score, jd_info, weights)


if __name__ == "__main__":
>>>>>>> d900ca65ebb4af9d8ce4e4aa049ebacf1695cc56
    main()