# Legixo — Legal Q&A HTTP API

A legal document Question-Answering HTTP API built with **Python 3.12**, **LangGraph**, **Pinecone Serverless**, and **Google Gemini AI**.

---

## Video Walkthrough & Demo

- **Submission Video**: [Link to 5–10 Min Video Walkthrough](YOUR_VIDEO_URL_HERE) _(Replace with your video link)_

---

## Quickstart Guide

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dhruvked/Lexixo-Assignment.git
cd Lexixo-Assignment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Environment Setup

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=legixo-corpus

# Optional — LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT="legixo-qa"
```

---

### 3. Document Ingestion

Run the ingestion script to process documents from `corpus/`, chunk them, generate embeddings (`gemini-embedding-001`), and index them in Pinecone:

```bash
python app/ingest.py
```

#### 🔄 Ingestion Idempotency & Overwrite Handling

- **Chunk ID Strategy**: Chunks are assigned deterministic IDs formatted as `{filename}_chunk_{idx}` (e.g. `02_employment_agreement_excerpt.md_chunk_0`).
- **Upsert Overwrite**: Pinecone `upsert()` operations are idempotent. Running ingestion multiple times overwrites existing vector records with matching IDs rather than creating duplicates.
- **Assumption**: Corpus filenames within the `corpus/` directory are unique.

---

### 4. Running the API Server

Start the FastAPI dev server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.

---

## 📡 API Usage & Endpoint Documentation

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

### Ask Question (`POST /ask`)

#### Example 1: In-Corpus Legal Question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement?"}'
```

**Response:**

```json
{
  "question": "What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement?",
  "answer": "Either party may end the employment agreement by giving 60 days written notice. During notice, the employee must hand over all laptops, badges, and source code access.",
  "citations": ["02_employment_agreement_excerpt.md"],
  "status": "success",
  "step_count": 3
}
```

#### Example 2: Out-of-Corpus Question (Refusal Path)

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the population of Riverside city?"}'
```

**Response:**

```json
{
  "question": "What is the population of Riverside city?",
  "answer": "I cannot find relevant information in the provided document set to answer your question.",
  "citations": [],
  "status": "success",
  "step_count": 3
}
```

---

##  Running Automated Evaluation

Run the evaluation test suite against 19 gold test cases (rule-based fact recall + LLM-as-judge faithfulness):

```bash
python eval/run_eval.py
```

Detailed test outputs are saved to [`eval/eval_results.json`](eval/eval_results.json).

---

##  Architecture & LangGraph Flow

For a full breakdown of graph nodes, state transitions, and Mermaid diagrams, see [`docs/langgraph.md`](docs/langgraph.md).

```
[Retrieve] ──> [Grade Documents] ──(Router)──> [Generate Answer] ──> [END]
                                       └──> [Fallback Answer] ──> [END]
```
