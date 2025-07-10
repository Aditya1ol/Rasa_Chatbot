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

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, filename="chatbot.log")

# Configuration
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
if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    logging.info(f"Loaded FAISS index and {len(chunks)} chunks")

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
def extract_chunks_from_pdf(pdf_path, max_chars=500):
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as doc:
            for page in doc.pages:
                text = page.extract_text() or ""
                text = clean_text(text)
                sentences = text.split('. ')
                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < max_chars:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                if current_chunk:
                    chunks.append(current_chunk.strip())
    except Exception as e:
        logging.error(f"Failed to process {pdf_path}: {e}")
    return chunks

# Process PDFs into FAISS index
def process_all_pdfs():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        logging.info(f"Using cached index '{INDEX_PATH}' and chunks '{CHUNKS_PATH}'")
        return
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            logging.info(f"Processing: {filename}")
            chunks = extract_chunks_from_pdf(pdf_path)
            all_chunks.extend(chunks)
    if all_chunks:
        logging.info(f"Total Chunks: {len(all_chunks)}")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        logging.info(f"Saved index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'")
    else:
        logging.warning("No PDF content found to index.")

# Generate T5 response
def generate_t5_response(message, context=None):
    cache_key = (message, tuple(context or []))
    if cache_key in cache:
        return cache[cache_key]

    # Preprocess context chunks to remove "Question X:" and "Answer:" prefixes for cleaner input
    def clean_chunk(chunk):
        chunk = re.sub(r'Question \d+:', '', chunk, flags=re.IGNORECASE)
        chunk = re.sub(r'Answer:', '', chunk, flags=re.IGNORECASE)
        return chunk.strip()

    cleaned_context = [clean_chunk(c) for c in context] if context else []
    context_text = " ".join(cleaned_context) if cleaned_context else "UIET Chandigarh is a premier engineering institute offering B.E., M.E., and Ph.D. programs."

    input_text = f"Context: {context_text} Question: {message} Answer:"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(
        inputs,
        max_length=150,
        num_beams=10,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        max_new_tokens=100
    )
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = ' '.join(response.split())
    if len(response) > 300:
        response = response[:300] + "..."
    cache[cache_key] = response

# Get Rasa response
def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message}, timeout=5)
        response.raise_for_status()
        return [msg.get("text", "") for msg in response.json() if msg.get("text")]
    except requests.exceptions.RequestException as e:
        logging.error(f"Rasa error: {e}")
        return []

# Get PDF response
def get_pdf_response(message, k=5):
    if not index or not chunks:
        return ["No PDF data found."]
    try:
        query_embedding = EMBEDDING_MODEL.encode([message])
        distances, indices = index.search(query_embedding, k * 3)  # Search more to filter later
        candidate_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]

        # Filter out redundant chunks based on semantic similarity
        filtered_chunks = []
        filtered_embeddings = []
        for chunk in candidate_chunks:
            chunk_embedding = EMBEDDING_MODEL.encode([chunk])[0]
            if not any(np.dot(chunk_embedding, fe) / (np.linalg.norm(chunk_embedding) * np.linalg.norm(fe)) > 0.85 for fe in filtered_embeddings):
                filtered_chunks.append(chunk)
                filtered_embeddings.append(chunk_embedding)
            if len(filtered_chunks) >= k:
                break

        pdf_response_state["chunks"] = filtered_chunks
        pdf_response_state["pointer"] = min(2, len(filtered_chunks))
        return filtered_chunks[:2]
    except Exception as e:
        logging.error(f"PDF search error: {e}")
        return ["Error processing PDF."]

# Deduplicate responses
def deduplicate_responses(responses, threshold=0.9):
    embeddings = EMBEDDING_MODEL.encode(responses)
    unique_responses = []
    for i, (resp, emb) in enumerate(zip(responses, embeddings)):
        if not any(util.cos_sim(emb, EMBEDDING_MODEL.encode([ur])) > threshold for ur in unique_responses):
            unique_responses.append(resp)
    return unique_responses

# Semantic FAQ matching
def find_faq_answer(user_message):
    user_embedding = EMBEDDING_MODEL.encode([user_message])
    faq_questions = [faq["question"].lower() for faq in saved_faq]
    faq_embeddings = EMBEDDING_MODEL.encode(faq_questions)
    similarities = util.cos_sim(user_embedding, faq_embeddings)[0]
    best_idx = similarities.argmax().item()
    if similarities[best_idx] > 0.8:
        return saved_faq[best_idx]["answer"]
    return None

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
        if pdf_response_state["chunks"]:
            start = pdf_response_state["pointer"]
            remaining_chunks = pdf_response_state["chunks"][start:]
            if remaining_chunks:
                next_chunks = remaining_chunks[:2]
                pdf_response_state["pointer"] += len(next_chunks)
                return jsonify({"response": next_chunks})
            else:
                return jsonify({"response": ["No more information to show."]})
        else:
            return jsonify({"response": ["No previous answer to continue from."]})
    
    # Check FAQ
    faq_answer = find_faq_answer(user_message)
    if faq_answer:
        return jsonify({"response": [faq_answer]})

    # Get responses
    rasa_responses = get_rasa_response(user_message)
    pdf_chunks = get_pdf_response(user_message)
    t5_response = generate_t5_response(user_message, context=pdf_chunks)

    # Combine responses
    final_responses = []
    if rasa_responses and t5_response and t5_response.strip() and not any(gf in t5_response.lower() for gf in generic_fallbacks):
        # Combine Rasa and T5 responses
        combined_response = rasa_responses[0].strip()
        if t5_response.strip() not in combined_response:
            combined_response += " " + t5_response.strip()
        final_responses.append(combined_response)
    elif rasa_responses:
        final_responses.extend(rasa_responses[:1])
    elif t5_response and t5_response.strip() and not any(gf in t5_response.lower() for gf in generic_fallbacks):
        final_responses.append(t5_response)
    elif pdf_chunks:
        final_responses.extend(pdf_chunks)
    else:
        final_responses.append("Sorry, I couldn't find any relevant information.")

    final_responses = deduplicate_responses(final_responses)
    return jsonify({"response": final_responses})

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
    process_all_pdfs()
    app.run(host="0.0.0.0", port=5000, debug=True)