from flask import Flask, render_template, request, jsonify
import os
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

# Track continuation state
pdf_response_state = {
    "chunks": [],
    "pointer": 0
}

# Configuration
PDF_FOLDER = "pdfs"
INDEX_PATH = "pdf_index.faiss"
CHUNKS_PATH = "pdf_chunks.pkl"
RASA_API_URL = "http://localhost:5005/webhooks/rest/webhook"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load pre-saved questions from college_faq.json
with open("college_faq.json", "r", encoding="utf-8") as f:
    saved_faq = json.load(f)

# Load generated questions from generated_questions.json
GENERATED_QUESTIONS_PATH = "generated_questions.json"
if os.path.exists(GENERATED_QUESTIONS_PATH):
    with open(GENERATED_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        generated_questions = json.load(f)
else:
    generated_questions = []

# 📄 Extract text chunks from a PDF
import re

def clean_text(text):
    # Remove multiple spaces, newlines, tabs
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    text = text.strip()
    return text

def extract_chunks_from_pdf(pdf_path, max_chars=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        # Clean extracted text
        text = clean_text(text)
        lines = text.split('. ')  # Split by sentences for better chunking
        current_chunk = ""
        for line in lines:
            if len(current_chunk) + len(line) < max_chars:
                current_chunk += line + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
    return chunks

# ⚙️ Process all PDFs into FAISS index
def process_all_pdfs():
    # Check if index and chunks already exist to avoid reprocessing
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print(f"✅ Using cached index '{INDEX_PATH}' and chunks '{CHUNKS_PATH}'")
        return

    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"🔍 Processing: {filename}")
            try:
                chunks = extract_chunks_from_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"❌ Failed to process {filename}: {e}")

    if all_chunks:
        print(f"💬 Total Chunks: {len(all_chunks)}")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        print(f"✅ Saved index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'")
    else:
        print("⚠️ No PDF content found to index.")

# 🤖 Get response from Rasa
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5TokenizerFast

# Load T5 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model_path = "actions/trained_model"
try:
    # Load model from PyTorch .bin file in actions/trained_model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_path).to(device)
except Exception as e:
    print(f"Failed to load model from {t5_model_path}: {e}")
    # Fallback: Load from new_college_bot with safetensors (may fail)
    t5_tokenizer = T5Tokenizer.from_pretrained("new_college_bot")
    t5_model = T5ForConditionalGeneration.from_pretrained("new_college_bot").to(device)

def generate_t5_response(message, context=None):
    # If context is provided, prepend it to the message for T5 input
    input_text = message
    if context:
        # Combine context chunks into a single string
        context_text = " ".join(context)
        input_text = f"Context: {context_text} Question: {message}"

    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = t5_model.generate(
        inputs,
        max_length=150,
        num_beams=10,
        early_stopping=True,
        no_repeat_ngram_size=3,  # Avoid repeating phrases
        length_penalty=2.0,      # Encourage shorter responses
        max_new_tokens=100,       # Limit generated tokens
        num_return_sequences=1,   # Generate 1 response to avoid confusion
        do_sample=False           # Disable sampling for deterministic output
    )
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    # Fix spacing issues by replacing multiple spaces with single space
    response = ' '.join(response.split())

    # Post-process to truncate overly long responses
    if len(response) > 300:
        response = response[:300] + "..."
    return response

def get_rasa_response(message):
    try:
        response = requests.post(RASA_API_URL, json={"sender": "user", "message": message})
        response.raise_for_status()
        bot_messages = response.json()
        return [msg.get("text", "") for msg in bot_messages if msg.get("text")]
    except requests.exceptions.RequestException as e:
        print(f"❌ Rasa error: {e}")
        return []

# 📚 Get response from PDF semantic search
def get_pdf_response(message, k=3):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return ["No PDF data found."]

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
            pdf_response_state["chunks"] = chunks

        query_embedding = EMBEDDING_MODEL.encode([message])
        distances, indices = index.search(query_embedding, k)
        results = [chunks[idx] for idx in indices[0] if idx < len(chunks)]

        pdf_response_state["pointer"] = min(2, len(results))

        return results[:2]
    except Exception as e:
        print(f"❌ PDF search error: {e}")
        return ["Error processing the PDF."]

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/chatbot")
def home():
    # Pass saved questions and generated questions to template
    return render_template("index.html", saved_questions=saved_faq, generated_questions=generated_questions)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()

    # CONTINUE case
    if user_message in ["continue", "more"]:
        if pdf_response_state["chunks"]:
            start = pdf_response_state["pointer"]
            end = start + 2
            next_chunks = pdf_response_state["chunks"][start:end]
            pdf_response_state["pointer"] = end
            if next_chunks:
                return jsonify({"response": next_chunks})
            else:
                return jsonify({"response": ["No more information to show."]})
        else:
            return jsonify({"response": ["There's no previous answer to continue from."]})

    # Check if user_message matches any FAQ question exactly (case-insensitive)
    matched_answer = None
    for faq_entry in saved_faq:
        faq_question = faq_entry.get("question", "").strip().lower()
        if user_message == faq_question:
            matched_answer = faq_entry.get("answer", "").strip()
            break

    if matched_answer:
        # Return exact FAQ answer without generalization
        return jsonify({"response": [matched_answer]})

    # New query case
    rasa_responses = get_rasa_response(user_message)

    # Get top 5 semantically matched chunks
    all_chunks = []
    top_chunks = []
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, "rb") as f:
            all_chunks = pickle.load(f)
        index = faiss.read_index(INDEX_PATH)
        query_embedding = EMBEDDING_MODEL.encode([user_message])
        distances, indices = index.search(query_embedding, 5)
        top_chunks = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]

    # Store chunks for continuation
    pdf_response_state["chunks"] = top_chunks
    pdf_response_state["pointer"] = 2  # first 2 shown immediately

    # Prepare response
    response_chunks = top_chunks[:2]

    # Generate T5 response with PDF context
    t5_response = generate_t5_response(user_message, context=response_chunks)

    # Combine responses with priority: Rasa for question answers, then T5, then PDF chunks
    final_responses = []
    # Add Rasa response if available
    if rasa_responses:
        for resp in rasa_responses[:1]:
            if resp.strip():
                final_responses.append(resp)
                break
    # Add T5 responses if available and different from Rasa
    elif t5_response:
        # Combine multiple T5 responses into one summary
        combined_response = " ".join([resp.strip() for resp in t5_response if resp.strip()])
        if combined_response:
            final_responses.append(combined_response)
    # Add PDF chunks only if not duplicate
    elif response_chunks:
        for chunk in response_chunks:
            if chunk.strip():
                final_responses.append(chunk)
                break

    if not final_responses:
        final_responses = ["Sorry, I couldn't find any relevant information."]

    return jsonify({"response": final_responses})

if __name__ == "__main__":
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)
    process_all_pdfs()
    app.run(host="0.0.0.0", port=5000, debug=True)
