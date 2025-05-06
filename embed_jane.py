import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

input_file = Path("data/patient8/patient8_chunks_readable.json")
output_file = Path("data/patient8/patient8_embedded_chunks.json")

with open(input_file, "r", encoding="utf-8") as f:
    structured_data = json.load(f)

embedded_chunks = []

def embed_from_structured(data):
    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                recurse(v, prefix + k.replace("_", " ").capitalize() + ": ")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                recurse(item, prefix + f"[{idx}] ")
        else:
            question = prefix.strip()
            answer = str(obj)
            if question and answer:
                text = f"{question} {answer}"
                try:
                    embedding = client.embeddings.create(
                        input=[text],
                        model="text-embedding-3-small"
                    ).data[0].embedding
                    embedded_chunks.append({"text": text, "embedding": embedding})
                except Exception as e:
                    print(f"Embedding failed for: {text[:50]}... Reason: {e}")

    recurse(data)

embed_from_structured(structured_data)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, indent=2)

print(f"✅ Embedded {len(embedded_chunks)} chunks to {output_file}")
