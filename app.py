from flask import Flask, render_template, request, jsonify
import os
import pdfplumber
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
import json
import requests
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging
from cachetools import TTLCache
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, filename="chatbot.log")

# Configuration
generic_fallbacks = [
    "sorry", "couldn't find", "no relevant information", "not sure", "don't know"
]
PDF_FOLDER = "pdfs"
INDEX_PATH = "pdf_index.faiss"
CHUNKS_PATH = "pdf_chunks.pkl"
RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005/webhooks/rest/webhook")
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
cache = TTLCache(maxsize=100, ttl=3600)

# Global T5 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("actions/trained_model")
    t5_model = T5ForConditionalGeneration.from_pretrained("actions/trained_model").to(device)
    logging.info("Loaded T5 model from actions/trained_model")
except Exception as e:
    logging.warning(f"Failed to load T5 model: {e}. Using t5-small.")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

# Global FAISS index
index = None
chunks = []

# Track continuation state
pdf_response_state = {
    "chunks": [],
    "pointer": 0
}

# Load FAQs
with open("college_faq.json", "r", encoding="utf-8") as f:
    saved_faq = json.load(f)

# Load generated questions
GENERATED_QUESTIONS_PATH = "generated_questions.json"
if os.path.exists(GENERATED_QUESTIONS_PATH):
    with open(GENERATED_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        generated_questions = json.load(f)
else:
    generated_questions = []

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extract chunks from PDF


def extract_chunks_from_pdf(pdf_path, window_size=3, stride=1):
    chunks = []
    try:
        print(f"\U0001F4C4 Extracting from: {pdf_path}")
        with pdfplumber.open(pdf_path) as doc:
            for page in doc.pages:
                text = page.extract_text() or ""
                text = clean_text(text)
                sentences = sent_tokenize(text)

                for i in range(0, len(sentences), stride):
                    chunk = " ".join(sentences[i:i + window_size])
                    if chunk:
                        chunks.append(chunk.strip())
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
    print(f"✅ Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks

# Process all PDFs into FAISS index
def process_all_pdfs():
    global index, chunks

    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print(f"✅ Using cached FAISS index from '{INDEX_PATH}'")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        print(f"✅ Loaded {len(chunks)} chunks from '{CHUNKS_PATH}'")
        return

    print("🔄 No cache found. Extracting chunks from PDFs...")
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            extracted = extract_chunks_from_pdf(pdf_path)
            all_chunks.extend(extracted)

    if all_chunks:
        print(f"✅ Extracted {len(all_chunks)} total chunks from PDFs.")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        chunks = all_chunks
        print(f"✅ Saved FAISS index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'")
    else:
        print("⚠️ No PDF content found to process.")

# T5 Response
def generate_t5_response(message, context=None):
    cache_key = (message, tuple(context or []))
    if cache_key in cache:
        return cache[cache_key]

    def clean_chunk(chunk):
        chunk = re.sub(r'Question \d+:', '', chunk, flags=re.IGNORECASE)
        chunk = re.sub(r'Answer:', '', chunk, flags=re.IGNORECASE)
        return chunk.strip()

    cleaned_context = [clean_chunk(c) for c in context] if context else []
    context_text = " ".join(cleaned_context) if cleaned_context else "UIET Chandigarh is a premier engineering institute."

    input_text = (
    f"Based on the document content below, answer the question clearly.\n"
    f"Context:\n{context_text}\n\n"
    f"Question: {message}\nAnswer:"
)

    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(
        inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    cache[cache_key] = response
    return response

# PDF Response
def get_pdf_response(message, k=5):
    if not index or not chunks:
        return ["No PDF data found."]
    
    query_embedding = EMBEDDING_MODEL.encode([message])
    distances, indices = index.search(query_embedding, k * 5)
    
    candidate_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    filtered = []
    used = []

    for chunk in candidate_chunks:
        chunk_emb = EMBEDDING_MODEL.encode([chunk])[0]
        if all(util.cos_sim(chunk_emb, u)[0][0] < 0.9 for u in used):
            filtered.append(chunk)
            used.append(chunk_emb)
        if len(filtered) >= k:
            break

    pdf_response_state["chunks"] = filtered
    pdf_response_state["pointer"] = min(2, len(filtered))
    return filtered[:2]



# FAQ Answer
def find_faq_answer(user_message, threshold=0.75):
    if not saved_faq:
        return None

    questions = [faq["question"] for faq in saved_faq]
    question_embeddings = EMBEDDING_MODEL.encode(questions)
    query_embedding = EMBEDDING_MODEL.encode(user_message)

    similarities = util.cos_sim(query_embedding, question_embeddings)[0]
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score >= threshold:
        return saved_faq[best_idx]["answer"]
    return None


# Rasa Response
def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message}, timeout=5)
        response.raise_for_status()
        return [msg.get("text", "") for msg in response.json() if msg.get("text")]
    except Exception as e:
        logging.error(f"Rasa error: {e}")
        return []

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/chatbot")
def home():
    return render_template("index.html", saved_questions=saved_faq, generated_questions=generated_questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    if user_message in ["continue", "more"]:
        start = pdf_response_state["pointer"]
        remaining_chunks = pdf_response_state["chunks"][start:]
        if remaining_chunks:
            next_chunks = remaining_chunks[:2]
            pdf_response_state["pointer"] += len(next_chunks)
            return jsonify({"response": next_chunks})
        else:
            return jsonify({"response": ["No more information to show."]})

    # ✅ Priority 1: FAQ
    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})  
    
    # ✅ Priority 3: Rasa first (small talk)
    rasa_responses = get_rasa_response(user_message)
    if rasa_responses:
        return jsonify({"response": rasa_responses})  
    
    # ✅ Priority 2: PDF chunks
    pdf_chunks = get_pdf_response(user_message)
    if pdf_chunks and "No PDF data found." not in pdf_chunks[0]:
        t5_response = generate_t5_response(user_message, pdf_chunks)

        if isinstance(t5_response, str) and t5_response.strip().lower() not in ["", "true", "false"]:
           return jsonify({"response": [t5_response]})

    # ❌ Fallback
    return jsonify({"response": ["Sorry, I couldn't find any relevant information."]})

@app.route("/save_faq", methods=["POST"])
def save_faq():
    new_faq = request.json.get("faq", {})
    saved_faq.append(new_faq)
    with open("college_faq.json", "w", encoding="utf-8") as f:
        json.dump(saved_faq, f, indent=2)
    return jsonify({"message": "FAQ saved successfully"})

if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
        print(f"📁 Created folder: {PDF_FOLDER}")
    process_all_pdfs()
    app.run(host="0.0.0.0", port=5000, debug=True)
