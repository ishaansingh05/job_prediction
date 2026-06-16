
# AI Resume Intelligence System (RAG-Based Job Matching Engine)

A production-style **AI-powered Resume-to-Job Matching System** built using a **Retrieval-Augmented Generation (RAG) pipeline**.

It combines:

- Semantic vector search (FAISS)
- Transformer embeddings (SentenceTransformers)
- LLM reasoning (Groq - LLaMA 3)
- Resume parsing pipeline
- Skill gap + career recommendation engine

---

# 📌 Live Demo
Upload a resume and instantly receive:
- Ranked job matches
- Skill gap insights
- AI-generated career guidance

---

# 🧠 System Overview (RAG Architecture)

This project implements a full **Retrieval-Augmented Generation pipeline**.


---

# 🧩 Key Components

## 1. Semantic Job Retrieval (FAISS)

The system uses **FAISS IndexFlatIP** for high-speed similarity search over job embeddings.

- Converts job descriptions into dense vectors
- Performs cosine similarity search
- Retrieves most relevant job matches instantly

---

## 2. RAG-Based Intelligence Layer

After retrieval, a **Large Language Model (Groq LLaMA 3 70B)** is used to:

- Rank job relevance
- Explain job-fit reasoning
- Identify matching skills
- Detect skill gaps
- Provide career guidance

---

## 3. Resume Processing Pipeline

Supports multiple formats:

- PDF resumes (PyMuPDF)
- DOCX resumes (python-docx)

Extracted information:

- Skills
- Experience
- Education
- Projects

---

## 4. Text Cleaning System

Before embedding, all text is normalized:

- HTML removal
- URL removal
- Bullet noise cleanup
- Whitespace normalization

This ensures **high-quality embeddings and better retrieval accuracy**.

---

## 5. Safe JSON Parsing Layer (Critical)

LLMs often return malformed JSON.  
To ensure production safety, a robust parser is implemented:

```python id="safejson1"
def safe_json_parse(text):
    try:
        return json.loads(text)

    except json.JSONDecodeError:

        cleaned = re.sub(r"```json|```", "", text).strip()

        try:
            return json.loads(cleaned)

        except json.JSONDecodeError:

            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())

            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group())

            raise ValueError("Invalid JSON from LLM")
---

## **Embedding Model**
The system uses the following Sentence Transformer model:
**sentence-transformers/all-MiniLM-L6-v2**
This model converts text into dense vector embeddings.

##⚙️ Project Structure
📦 Resume-Job-Matching-RAG
│
├── create.ipynb               # Main pipeline (data prep + FAISS + LLM)
├── jobs_app.py               # Streamlit/Gradio app UI
├── requirements.txt          # Dependencies
│
├── jobs_index               # FAISS vector index (large binary file)
├── jobs_meta.pkl            # Job metadata (large pickle file)
│
├── assets/                  # Images for README
│   ├── architecture.png
│   ├── demo.png

##⚠️ Large File Handling
Due to size constraints, the following files are NOT included in GitHub:

jobs_index
jobs_meta.pkl
These are generated locally using:
**create.ipynb**
They can be rebuilt anytime from the dataset.

##⚙️ Tech Stack
Python 3.10
FAISS (Vector Database)
SentenceTransformers
Groq API (LLaMA 3)
PyMuPDF
Pandas / NumPy
Streamlit

##📊 Dataset

A real-world job dataset containing ~97,000 records:

Job title
Company name
Skills
Job description
Experience level
Location

Each record is transformed into structured text before embedding.
