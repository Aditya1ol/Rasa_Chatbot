import os
from extract_pdf import extract_chunks_from_pdf_pages

def save_chunks_to_file(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\\n\\n")

def main():
    pdf1 = "pdfs/IT syllabus.pdf"
    pdf2 = "pdfs/20220823162407-b.e.eee1stto8thsem.forthebatch2022-26 - 2025-04-14T001047.826.pdf"

    # Extract pages 1-8 from IT syllabus.pdf
    chunks1 = extract_chunks_from_pdf_pages(pdf1, 1, 8)
    save_chunks_to_file(chunks1, "extracted_it_syllabus_initial.txt")
    print(f"Extracted {len(chunks1)} chunks from {pdf1} pages 1-8")

    # Extract pages 1-15 from the other PDF
    chunks2 = extract_chunks_from_pdf_pages(pdf2, 1, 15)
    save_chunks_to_file(chunks2, "extracted_22file_initial.txt")
    print(f"Extracted {len(chunks2)} chunks from {pdf2} pages 1-15")

if __name__ == "__main__":
    main()
