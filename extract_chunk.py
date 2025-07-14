import os
import pdfplumber
import re
import json

PDF_FOLDER = "pdfs"
MAX_CHARS = 500

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_chunks_from_pdf(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as doc:
        for page in doc.pages:
            text = page.extract_text() or ""
            text = clean_text(text)
            sentences = text.split('. ')
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) < MAX_CHARS:
                    chunk += sent + ". "
                else:
                    if chunk:
                        chunks.append(chunk.strip())
                    chunk = sent + ". "
            if chunk:
                chunks.append(chunk.strip())
    return chunks

all_chunks = []
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        all_chunks.extend(extract_chunks_from_pdf(path))

print(f"Extracted {len(all_chunks)} chunks.")
