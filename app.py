from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()
app = FastAPI()

class Query(BaseModel):
    patient_id: str
    question: str

@app.post("/ask")
def ask(query: Query):
    messages = [
        {"role": "system", "content": "You are a helpful medical assistant. Answer medical questions based on your general knowledge. Do not engage in any other conversation."},
        {"role": "user", "content": query.question}
    ]

    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return {"answer": res.choices[0].message.content}
