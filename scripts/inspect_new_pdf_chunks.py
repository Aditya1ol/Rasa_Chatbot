import sys
sys.path.append("..")
from extract_pdf import extract_chunks_from_pdf

pdf_path = "pdfs/querry 6th sem.pdf"

chunks = extract_chunks_from_pdf(pdf_path)
print(f"Extracted {len(chunks)} chunks from {pdf_path}")

for i, chunk in enumerate(chunks[:10]):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
