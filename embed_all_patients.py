import os
import json
from openai import OpenAI
from pathlib import Path
from datetime import datetime

client = OpenAI()

DATA_DIR = Path("data")
ERROR_LOG = Path("embedding_errors.log")

def log_error(patient_id, message):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {patient_id}: {message}\n")

for patient_dir in DATA_DIR.glob("patient*"):
    patient_id = patient_dir.name
    chunks_file = patient_dir / f"{patient_dir.name}_chunks.json"
    out_file = patient_dir / f"{patient_dir.name}_embedded_chunks.json"

    if not chunks_file.exists():
        msg = "chunks.json not found."
        print(f"⚠️ Skipping {patient_id}: {msg}")
        log_error(patient_id, msg)
        continue

    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as e:
        msg = f"Failed to load chunks.json: {e}"
        print(f"❌ {patient_id}: {msg}")
        log_error(patient_id, msg)
        continue

    embedded_chunks = []
    for idx, chunk in enumerate(chunks):
        text = chunk.strip() if isinstance(chunk, str) else chunk.get("text", "").strip()
        if not text:
            continue
        try:
            embedding = client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            ).data[0].embedding

            embedded_chunks.append({
                "text": text,
                "embedding": embedding
            })
        except Exception as e:
            msg = f"Chunk {idx} failed to embed: {e}"
            print(f"❌ {patient_id}: {msg}")
            log_error(patient_id, msg)

    if embedded_chunks:
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(embedded_chunks, f)
            print(f"✅ Embedded {len(embedded_chunks)} chunks for {patient_id}")
        except Exception as e:
            msg = f"Failed to write embedded_chunks.json: {e}"
            print(f"❌ {patient_id}: {msg}")
            log_error(patient_id, msg)
    else:
        msg = "No valid chunks embedded."
        print(f"⚠️ {patient_id}: {msg}")
        log_error(patient_id, msg)
