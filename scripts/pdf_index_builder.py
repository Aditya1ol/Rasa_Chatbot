import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from extract_pdf import extract_chunks_from_pdf

PDF_FOLDER = "pdfs"
INDEX_PATH = "pdf_index.faiss"
CHUNKS_PATH = "pdf_chunks.pkl"
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_index():
    all_chunks = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            print(f"Processing: {filename}")
            try:
                chunks = extract_chunks_from_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    if all_chunks:
        print(f"Total chunks: {len(all_chunks)}")
        embeddings = EMBEDDING_MODEL.encode(all_chunks, show_progress_bar=True)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))

        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(all_chunks, f)
        print(f"Saved index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'")
    else:
        print("No PDF content found to index.")

if __name__ == "__main__":
    build_faiss_index()
