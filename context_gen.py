from sentence_transformers import SentenceTransformer, util
import json
import pickle
import os
import re

# Load cleaned FAQs
with open("augmented_data.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

# Load extracted PDF chunks
with open("pdf_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all PDF chunks
print("Embedding PDF chunks...")
chunk_embeddings = model.encode(chunks, convert_to_tensor=True)

# Helper: clean embedded answers from questions (if any)
def clean_question(text):
    return re.split(r'answer\s*:', text, flags=re.IGNORECASE)[0].strip()

# Build context-augmented QA dataset
qa_with_context = []
for item in faq_data:
    question = clean_question(item["question"])
    answer = item["answer"]

    # Encode question
    q_embedding = model.encode(question, convert_to_tensor=True)

    # Find most similar chunk from PDF
    similarities = util.pytorch_cos_sim(q_embedding, chunk_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_context = chunks[best_idx]

    qa_with_context.append({
        "context": best_context,
        "question": question,
        "answer": answer
    })

# Save to file
os.makedirs("data", exist_ok=True)
with open("data/uiet_qa.json", "w", encoding="utf-8") as f:
    json.dump(qa_with_context, f, indent=2)

print(f"Saved {len(qa_with_context)} context-augmented Q&A pairs to data/uiet_qa.json")
