import os
import fitz  # PyMuPDF
import re

def extract_chunks_from_pdf(pdf_path):
    """
    Extracts text chunks from a PDF file by splitting on 'Question X:' pattern,
    so each chunk corresponds to one question and its answer(s).
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # Split text by 'Question X:' pattern, keep the delimiter by using lookahead
    chunks = re.split(r'(?=Question\s*\d+:)', full_text)

    # Remove empty or whitespace-only chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # If the first chunk does not start with 'Question', it might be preamble, discard it
    if chunks and not chunks[0].lower().startswith('question'):
        chunks = chunks[1:]

    return chunks

def extract_chunks_from_pdf_pages(pdf_path, start_page, end_page):
    """
    Extracts text chunks from a PDF file for pages in the range [start_page, end_page],
    splitting by 'Question X:' pattern.
    Pages are 1-indexed.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page_num in range(start_page - 1, min(end_page, len(doc))):
        page = doc[page_num]
        full_text += page.get_text() + "\n"

    chunks = re.split(r'(?=Question\s*\d+:)', full_text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if chunks and not chunks[0].lower().startswith('question'):
        chunks = chunks[1:]
    return chunks

if __name__ == "__main__":
    PDF_FOLDER = "pdfs"
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing: {filename}")
            try:
                chunks = extract_chunks_from_pdf(pdf_path)
                print(f"Extracted {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
