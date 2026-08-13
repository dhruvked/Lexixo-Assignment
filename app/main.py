from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.graph import graph_app

app = FastAPI(title = "Legixo Q&A API")

class QuestionRequest(BaseModel):
    question:str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(payload: QuestionRequest):
    user_question = payload.question

    if not user_question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

    initial_state = {
        "question": user_question,
        "documents": [],
        "is_relevant": False,
        "answer": "",
        "citations": [],
        "step_count": 0
    }

    final_state = graph_app.invoke(initial_state)
    return {
        "question": user_question,
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "documents": final_state["documents"], 
        "status": "success",
        "step_count": final_state["step_count"]
    }