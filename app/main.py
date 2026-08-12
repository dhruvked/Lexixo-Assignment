from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "Legixo Q&A API")

class QuestionRequest(BaseModel):
    question:str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/ask")
def ask(payload: QuestionRequest):
    user_question = payload.question

    return{
        "question":user_question,
        "answer": f"You asked : '{user_question}'. Langraph flow here",
        "citation": []
    }