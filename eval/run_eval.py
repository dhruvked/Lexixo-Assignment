"""
Legixo Q&A - Evaluation Script
================================
Runs all test cases from evaluations.json against the live /ask API.

Metrics:
  In-Corpus  → Fact Recall, Citation Accuracy, Faithfulness (LLM judge)
  Out-of-Corpus → Refusal Rate, Hallucination Check (LLM judge)
  All        → Latency (seconds per request)

Usage:
  1. Start the API server: uvicorn app.main:app --reload
  2. Run this script:      python eval/run_eval.py
"""

import os
import sys
import json
import time
import re

import requests
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

API_URL        = "http://127.0.0.1:8000/ask"
EVAL_FILE      = "evaluations.json"  
RESULTS_FILE   = "eval_results.json"
LLM_MODEL      = "gemini-3.1-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Delay between test cases (seconds) — set to 30s to respect free-tier rate limits
REQUEST_DELAY  = 30

# Max retries on 429 rate limit errors
MAX_RETRIES    = 3

# Minimum fraction of expected facts that must appear for an in-corpus PASS
FACT_RECALL_THRESHOLD = 0.60

# Fallback phrases that count as a valid refusal for out-of-corpus questions
REFUSAL_PHRASES = [
    "cannot find",
    "not find",
    "do not have",
    "not in the",
    "unable to find",
    "no information",
    "not enough",
    "not sufficient",
]

# ---------------------------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating a legal Q&A system.

Question:
{question}

Retrieved Document Context:
{docs_context}

System Answer:
{answer}

Is the answer fully grounded in the retrieved document context above?
- Score 3: The answer uses ONLY information from the documents. No invented facts.
- Score 2: The answer is mostly grounded but adds minor assumptions or paraphrasing.
- Score 1: The answer invents facts that are NOT present in the documents (hallucination).

Reply with ONLY a single digit: 1, 2, or 3."""


def llm_faithfulness_score(question: str, docs: list, answer: str) -> int:
    """Ask Gemini to score answer faithfulness (1-3) with retry on 429."""
    if not answer or answer.strip() == "":
        return 1

    docs_context = "\n\n".join([
        f"Source: {d['filename']}\nContent: {d['text']}"
        for d in docs
    ]) if docs else "No documents retrieved."

    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        docs_context=docs_context,
        answer=answer
    )

    ai_client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ai_client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            digit = response.text.strip()[0]
            if digit in ("1", "2", "3"):
                return int(digit)
        except Exception as e:
            err = str(e)
            # Extract retry delay from error message if present
            match = re.search(r"retry in (\d+\.?\d*)s", err)
            wait = float(match.group(1)) + 2 if match else (attempt * 5)
            if "429" in err and attempt < MAX_RETRIES:
                print(f"\n    [LLM Judge] Rate limited. Waiting {wait:.0f}s before retry {attempt}/{MAX_RETRIES}...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    [LLM Judge Error] {err}")
                break

    return 1  # Default to worst score on failure


# ---------------------------------------------------------------------------
# Rule-Based Checks
# ---------------------------------------------------------------------------

def check_fact_recall(answer: str, expected_facts: list) -> dict:
    """Check which expected facts appear in the answer.

    Uses word-level matching: all words in the fact phrase must appear
    somewhere in the answer (handles bold markdown, reordering, etc.).
    Example: '60 days written notice' passes if '60', 'days', 'written',
    and 'notice' all appear anywhere in the answer.
    """
    answer_lower = answer.lower()
    results = {}
    for fact in expected_facts:
        words = fact.lower().split()
        results[fact] = all(word in answer_lower for word in words)
    return results


def check_citation(citations: list, expected_files: list) -> bool:
    """Check if at least one expected source file appears in citations."""
    return any(f in citations for f in expected_files)


def check_refusal(answer: str) -> bool:
    """Check if the answer correctly refuses to answer."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def divider(char="=", width=62):
    print(char * width)


def call_api(question: str) -> tuple[dict, float]:
    """Call /ask and return (response_json, elapsed_seconds) with retry on 429."""
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        resp = requests.post(API_URL, json={"question": question}, timeout=90)
        elapsed = round(time.time() - start, 2)
        if resp.status_code == 429:
            wait = attempt * 15
            print(f"  [API] Rate limited. Waiting {wait}s before retry {attempt}/{MAX_RETRIES}...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json(), elapsed
    resp.raise_for_status()  # Raise on final failed attempt


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def run_eval():
    # Load test cases
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total = len(test_cases)
    passed = 0
    failed = 0

    all_latencies        = []
    in_corpus_results    = []
    out_of_corpus_results = []

    divider()
    print(f"  LEGIXO Q&A — EVALUATION REPORT")
    print(f"  {total} test cases  |  API: {API_URL}")
    divider()
    print()

    # ------------------------------------------------------------------
    # Run each test case
    # ------------------------------------------------------------------
    for case in test_cases:
        q_id     = case["id"]
        question = case["question"]
        q_type   = case["type"]

        print(f"[Test {q_id}] ({q_type})")
        print(f"  Q: {question}")

        # Call API
        try:
            data, elapsed = call_api(question)
        except Exception as e:
            print(f"  ❌ API ERROR: {e}\n")
            failed += 1
            continue

        answer    = data.get("answer", "")
        citations = data.get("citations", [])
        api_docs  = data.get("documents", [])

        print(f"  A: {answer[:130]}{'...' if len(answer) > 130 else ''}")
        print(f"  Citations: {citations}")
        print(f"  Latency:   {elapsed}s")
        all_latencies.append(elapsed)

        # --------------------------------------------------------------
        # IN-CORPUS EVALUATION
        # --------------------------------------------------------------
        if q_type == "in_corpus":
            expected_facts = case.get("expected_facts", [])
            expected_files = case.get("expected_source_files", [])

            # 1. Fact Recall
            fact_hits = check_fact_recall(answer, expected_facts)
            n_hit     = sum(fact_hits.values())
            n_total   = len(expected_facts)
            recall_pct = round(n_hit / n_total * 100) if n_total else 0

            # 2. Citation Accuracy
            citation_ok = check_citation(citations, expected_files)

            # 3. LLM Faithfulness (1–3)
            print("  [LLM Judge] Scoring faithfulness...", end=" ", flush=True)
            faith_score = llm_faithfulness_score(question, api_docs, answer)
            faith_label = {1: "Hallucinated", 2: "Partially Grounded", 3: "Fully Grounded"}[faith_score]
            print(f"{faith_score}/3 ({faith_label})")

            # Pass condition: ≥60% facts + correct citation + faithfulness ≥ 2
            case_passed = (
                recall_pct >= (FACT_RECALL_THRESHOLD * 100)
                and citation_ok
                and faith_score >= 2
            )

            status = "✅ PASS" if case_passed else "❌ FAIL"
            print(f"  Fact Recall:      {n_hit}/{n_total} ({recall_pct}%)")
            print(f"  Citation Match:   {'✅' if citation_ok else '❌'}  (expected: {expected_files})")
            print(f"  Faithfulness:     {faith_score}/3  {faith_label}")
            print(f"  Result: {status}")

            in_corpus_results.append({
                "id": q_id,
                "question": question,
                "answer_snippet": answer[:300],
                "citations": citations,
                "fact_recall_pct": recall_pct,
                "fact_hits": fact_hits,
                "citation_ok": citation_ok,
                "faithfulness": faith_score,
                "latency_s": elapsed,
                "passed": case_passed
            })

        # --------------------------------------------------------------
        # OUT-OF-CORPUS EVALUATION
        # --------------------------------------------------------------
        else:
            # 1. Refusal check
            correctly_refused = check_refusal(answer)

            # 2. LLM hallucination check
            print("  [LLM Judge] Checking for hallucination...", end=" ", flush=True)
            faith_score = llm_faithfulness_score(question, api_docs, answer)
            no_hallucination = faith_score >= 2
            print(f"{faith_score}/3")

            # Pass: correctly refused AND didn't hallucinate
            case_passed = correctly_refused and no_hallucination

            status = "✅ PASS" if case_passed else "❌ FAIL"
            print(f"  Correctly Refused:  {'✅' if correctly_refused else '❌'}")
            print(f"  No Hallucination:   {'✅' if no_hallucination else '❌'}  (score {faith_score}/3)")
            print(f"  Result: {status}")

            out_of_corpus_results.append({
                "id": q_id,
                "question": question,
                "answer_snippet": answer[:300],
                "correctly_refused": correctly_refused,
                "faithfulness": faith_score,
                "latency_s": elapsed,
                "passed": case_passed
            })

        if case_passed:
            passed += 1
        else:
            failed += 1

        print()
        # Pause between test cases to respect free-tier rate limits
        time.sleep(REQUEST_DELAY)

    # ------------------------------------------------------------------
    # Aggregate Summary
    # ------------------------------------------------------------------
    divider()
    print("  SUMMARY")
    divider()

    # In-corpus aggregates
    if in_corpus_results:
        avg_recall    = round(sum(r["fact_recall_pct"] for r in in_corpus_results) / len(in_corpus_results), 1)
        citations_ok  = sum(1 for r in in_corpus_results if r["citation_ok"])
        avg_faith_ic  = round(sum(r["faithfulness"] for r in in_corpus_results) / len(in_corpus_results), 2)
        ic_passed     = sum(1 for r in in_corpus_results if r["passed"])

        print(f"\nIn-Corpus ({len(in_corpus_results)} questions):")
        print(f"  Fact Recall (avg):    {avg_recall}%")
        print(f"  Citation Accuracy:    {citations_ok}/{len(in_corpus_results)}")
        print(f"  Faithfulness (avg):   {avg_faith_ic}/3")
        print(f"  Passed:               {ic_passed}/{len(in_corpus_results)}")

    # Out-of-corpus aggregates
    if out_of_corpus_results:
        refusals     = sum(1 for r in out_of_corpus_results if r["correctly_refused"])
        avg_faith_oc = round(sum(r["faithfulness"] for r in out_of_corpus_results) / len(out_of_corpus_results), 2)
        oc_passed    = sum(1 for r in out_of_corpus_results if r["passed"])

        print(f"\nOut-of-Corpus ({len(out_of_corpus_results)} questions):")
        print(f"  Correct Refusals:     {refusals}/{len(out_of_corpus_results)}")
        print(f"  Faithfulness (avg):   {avg_faith_oc}/3")
        print(f"  Passed:               {oc_passed}/{len(out_of_corpus_results)}")

    # Latency summary
    if all_latencies:
        avg_lat  = round(sum(all_latencies) / len(all_latencies), 2)
        min_lat  = min(all_latencies)
        max_lat  = max(all_latencies)
        slowest  = test_cases[all_latencies.index(max_lat)]["question"][:60]

        print(f"\nLatency:")
        print(f"  Average:   {avg_lat}s")
        print(f"  Fastest:   {min_lat}s")
        print(f"  Slowest:   {max_lat}s  ← \"{slowest}...\"")

    # Overall
    overall_pct = round(passed / total * 100)
    print(f"\nOverall Pass Rate:  {passed}/{total}  ({overall_pct}%)")
    divider()

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    all_results = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "overall_pct": overall_pct,
            "avg_latency_s": round(sum(all_latencies) / len(all_latencies), 2) if all_latencies else None,
        },
        "in_corpus": in_corpus_results,
        "out_of_corpus": out_of_corpus_results,
    }

    os.makedirs("eval", exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved → {RESULTS_FILE}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick health check before running
    try:
        health = requests.get("http://127.0.0.1:8000/health", timeout=5)
        health.raise_for_status()
    except Exception:
        print("❌ ERROR: API server is not running.")
        print("   Start it with: uvicorn app.main:app --reload")
        sys.exit(1)

    run_eval()
