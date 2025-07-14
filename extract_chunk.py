import os
import fitz  # PyMuPDF
import re
import json
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

PDF_FOLDER = "pdfs"
MAX_CHARS = 500

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks_from_pdf(pdf_path, window_size=3, stride=1):
    chunks = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        text = clean_text(text)
        sentences = sent_tokenize(text)
        for i in range(0, len(sentences) - window_size + 1, stride):
            chunk = " ".join(sentences[i:i + window_size])
            if len(chunk) > 30:  # Optional: Skip tiny chunks
                chunks.append(chunk.strip())
    return chunks

# Collect all chunks
all_chunks = []
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        extracted = extract_chunks_from_pdf(path)
        all_chunks.extend(extracted)

print(f"✅ Extracted {len(all_chunks)} chunks.")
