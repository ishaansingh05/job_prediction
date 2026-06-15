import os
import json
import pickle
import re
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import faiss
import fitz
from sentence_transformers import SentenceTransformer
import pandas as pd

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    layout="wide",
    page_title="AI Resume → Career Copilot",
    page_icon="🚀"
)

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# CACHED MODELS
# ======================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=True)
def load_faiss_index_and_metadata():
    index = faiss.read_index(os.path.join(WORKDIR, "jobs.index"))
    with open(os.path.join(WORKDIR, "jobs_meta.pkl"), "rb") as f:
        job_meta = pickle.load(f)
    return index, job_meta


# ======================
# RESUME PROCESSING
# ======================
def extract_resume_text(file_bytes: bytes, file_name: str) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[•●▪◆►■]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ======================
# CORE SEARCH LOGIC (UNCHANGED)
# ======================
def search_jobs(resume_text, top_k, model, index, job_meta):

    query = resume_text  # IMPORTANT: unchanged core logic improved only by removing prefix

    resume_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32").reshape(1, -1)

    scores, indices = index.search(resume_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(job_meta):
            continue

        job = job_meta[int(idx)]
        results.append({
            "score": round(float(score), 4),
            "title": job["title"],
            "company": job["companyName"],
            "location": job["location"],
            "experience": job["experience"],
            "skills": job["tagsAndSkills"],
            "description": job["jobDescription"],
        })

    return results


# ======================
# GROQ (UNCHANGED LOGIC)
# ======================
def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return None


def extract_llm_analysis(resume_text, top_jobs, api_key):
    if not api_key:
        return None

    from groq import Groq
    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert recruiter.

Return ONLY valid JSON analysis of best matching jobs.

Resume:
{resume_text}

Jobs:
{json.dumps(top_jobs, indent=2)}

Return format:
[
 {{"title":"","company":"","match_score":0,"fit_level":"","why_match":"","matching_skills":[],"missing_skills":[],"career_advice":""}}
]
"""

    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return safe_json_parse(res.choices[0].message.content)


# ======================
# UI HELPERS
# ======================
def skill_gap_analysis(resume_text, job):
    resume_words = set(resume_text.lower().split())
    skills = str(job.get("skills", "")).lower().split(",")

    matched = [s.strip() for s in skills if s.strip() in resume_words]
    missing = [s.strip() for s in skills if s.strip() not in resume_words]

    return matched[:10], missing[:10]


def job_card(job, i):
    with st.container():
        st.markdown(f"""
        ### {job['title']}
        **🏢 {job['company']}**  
        📍 {job['location']} | 🎯 Score: {job['score']}
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Skills**")
            st.write(job["skills"][:200])

        with col2:
            st.markdown("**Description**")
            st.write(job["description"][:250])

        st.progress(min(float(job["score"]), 1.0))
        st.divider()


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.title("⚙️ Controls")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    top_k = st.slider("Top Jobs", 5, 20, 10)

    st.markdown("---")
    groq_key = st.text_input("Groq API Key", type="password")

# ======================
# MAIN UI
# ======================
st.title("🚀 AI Resume → Career Copilot")
st.caption("FAISS-powered semantic job matching system")

model = load_embedding_model()
index, job_meta = load_faiss_index_and_metadata()

if not uploaded_file:
    st.info("Upload your resume to start matching jobs.")
    st.stop()

# STEP 1
with st.spinner("📄 Reading resume..."):
    text = extract_resume_text(uploaded_file.read(), uploaded_file.name)
    clean = clean_text(text)

# STEP 2
with st.spinner("🧠 Encoding & searching jobs..."):
    jobs = search_jobs(clean, top_k, model, index, job_meta)

if not jobs:
    st.warning("No jobs found.")
    st.stop()

# ======================
# JOB RESULTS
# ======================
st.subheader("💼 Top Job Matches")

for i, job in enumerate(jobs):
    job_card(job, i)

# ======================
# DOWNLOAD
# ======================
df = pd.DataFrame(jobs)
st.download_button(
    "⬇️ Download Results CSV",
    df.to_csv(index=False),
    file_name="job_matches.csv",
    mime="text/csv"
)

# ======================
# LLM INSIGHTS
# ======================
if groq_key:
    with st.spinner("🧠 Generating AI insights..."):
        insights = extract_llm_analysis(clean, jobs, groq_key)

    if insights:
        st.subheader("🧠 AI Career Insights")

        for item in insights:
            with st.expander(f"{item.get('title')} - {item.get('company')}"):
                st.write("**Why Match:**", item.get("why_match"))
                st.write("**Missing Skills:**", item.get("missing_skills"))
                st.write("**Advice:**", item.get("career_advice"))