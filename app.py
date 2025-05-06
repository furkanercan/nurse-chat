from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import json
from pathlib import Path

client = OpenAI()
app = FastAPI()

class Query(BaseModel):
    patient_id: str
    question: str

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

@app.post("/ask")
def ask(query: Query):
    chunk_path = Path(f"data/{query.patient_id}/{query.patient_id}_embedded_chunks.json")
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Patient not found")

    with open(chunk_path, "r") as f:
        chunks = json.load(f)

    q_emb = get_embedding(query.question)
    top = sorted(
        chunks,
        key=lambda c: cosine_similarity(c["embedding"], q_emb),
        reverse=True
    )[:3]

    context = "\n\n".join([c["text"] for c in top])
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. If the question is based on the patient, then answer based only on the patient record. If the question is a general medical question, then answer it based on your general knowledge. Do not engage in any other conversation."},
        {"role": "user", "content": f"Patient Record:\n{context}\n\nQuestion: {query.question}"}
    ]

    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return {"answer": res.choices[0].message.content}
