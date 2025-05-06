🧪 To Run Locally (Without Docker)
bash

cd nursegame-rag-backend
pip install -r requirements.txt
uvicorn app:app --reload

🐳 To Run with Docker
bash

cd nursegame-rag-backend
docker build -t nursegame-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... nursegame-rag

You can then hit:

http://localhost:8000/ask