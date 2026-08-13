import os
from typing import TypedDict
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legixo-corpus")

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash"

# --- STATE ---
class GraphState(TypedDict):
    question: str
    documents: list
    is_relevant: bool
    answer: str
    citations: list
    step_count: int


# --- NODE 1: RETRIEVE ---
def retrieve_node(state: GraphState):
    question = state["question"]
    step_count = state.get("step_count", 0) + 1  # Fixed: was state.get(["step_count"], 0)

    print(f"\n[Node 1: Retrieve] Searching Pinecone for: '{question}'")

    ai_client = genai.Client(api_key=GEMINI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    res = ai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question
    )
    question_vector = res.embeddings[0].values

    results = index.query(
        vector=question_vector,
        top_k=4,
        include_metadata=True
    )

    documents = []
    for match in results["matches"]:
        meta = match["metadata"]
        documents.append({
            "text": meta.get("text", ""),
            "filename": meta.get("filename", "unknown"),
            "score": match["score"]
        })

    print(f"[Node 1: Retrieve] Found {len(documents)} chunks:")
    for doc in documents:
        print(f"  - {doc['filename']} (score: {doc['score']:.3f})")

    return {
        "documents": documents,
        "step_count": step_count
    }


# --- NODE 2: GRADE DOCUMENTS ---
def grade_documents_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    step_count = state.get("step_count", 0) + 1

    print(f"\n[Node 2: Grade] Evaluating {len(documents)} retrieved chunks...")

    if not documents:
        print("[Node 2: Grade] No documents retrieved. Marking as not relevant.")
        return {"is_relevant": False, "step_count": step_count}

    ai_client = genai.Client(api_key=GEMINI_API_KEY)

    docs_text = "\n\n".join([
        f"Source: {d['filename']}\nContent: {d['text']}"
        for d in documents
    ])

    prompt = f"""You are a document relevance evaluator.

Question: {question}

Retrieved Document Snippets:
{docs_text}

Do these document snippets contain enough relevant information to answer the question?
Reply with ONLY one word: YES or NO."""

    response = ai_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    reply = response.text.strip().upper()
    is_relevant = "YES" in reply

    print(f"[Node 2: Grade] Gemini says: '{reply}' -> is_relevant = {is_relevant}")

    return {
        "is_relevant": is_relevant,
        "step_count": step_count
    }


# --- NODE 3: GENERATE ANSWER ---
def generate_answer_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    step_count = state.get("step_count", 0) + 1

    print(f"\n[Node 3: Generate] Writing grounded answer...")

    ai_client = genai.Client(api_key=GEMINI_API_KEY)

    docs_context = "\n\n".join([
        f"Source: {d['filename']}\nContent: {d['text']}"
        for d in documents
    ])

    prompt = f"""You are a precise legal assistant. Answer the user's question using ONLY the document context below.
Do not use any outside knowledge. If the answer cannot be found in the context, say so.

Question: {question}

Document Context:
{docs_context}

Write a concise answer. At the very end, on a new line, write exactly:
SOURCES USED: filename1.md, filename2.md
Only list the filenames you actually quoted or referenced in your answer."""

    response = ai_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    raw = response.text.strip()

    # Parse citations from LLM response
    if "SOURCES USED:" in raw:
        parts = raw.split("SOURCES USED:")
        answer = parts[0].strip()
        citations = [f.strip() for f in parts[1].split(",") if f.strip()]
    else:
        # Fallback: use all retrieved filenames if LLM didn't follow format
        answer = raw
        citations = list(set([d["filename"] for d in documents]))

    print(f"[Node 3: Generate] Answer generated. Citations: {citations}")

    return {
        "answer": answer,
        "citations": citations,
        "step_count": step_count
    }


# --- NODE 4: FALLBACK ANSWER ---
def fallback_answer_node(state: GraphState):
    step_count = state.get("step_count", 0) + 1

    print(f"\n[Node 4: Fallback] No relevant documents found. Returning fallback.")

    return {
        "answer": "I cannot find relevant information in the provided document set to answer your question.",
        "citations": [],
        "step_count": step_count
    }


# --- ROUTER FUNCTION ---
def decide_to_generate(state: GraphState):
    # Guardrail: force fallback if too many steps
    if state.get("step_count", 0) > 5:
        print("[Router] Max steps exceeded. Routing to fallback.")
        return "fallback_answer"

    if state.get("is_relevant", False):
        return "generate_answer"
    else:
        return "fallback_answer"


# --- ASSEMBLE & COMPILE GRAPH ---
def build_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("fallback_answer", fallback_answer_node)

    # Entry point
    workflow.set_entry_point("retrieve")

    # Fixed edges
    workflow.add_edge("retrieve", "grade_documents")

    # Conditional branching from grade_documents
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "fallback_answer": "fallback_answer"
        }
    )

    # Both outcome nodes end the graph
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("fallback_answer", END)

    return workflow.compile()


# Compile once at import time so FastAPI can reuse it
graph_app = build_graph()