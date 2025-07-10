import json
import update_college_faq

def read_chunks_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    # Split chunks by double newlines
    chunks = [chunk.strip() for chunk in content.split("\\n\\n") if chunk.strip()]
    return chunks

def save_qa_pairs(qa_pairs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

def main():
    files = [
        "extracted_it_syllabus_initial.txt",
        "extracted_22file_initial.txt"
    ]
    all_chunks = []
    for file in files:
        chunks = read_chunks_from_file(file)
        all_chunks.extend(chunks)

    print(f"Read {len(all_chunks)} chunks from extracted text files.")

    qa_pairs = update_college_faq.parse_qa_pairs(all_chunks)
    print(f"Parsed {len(qa_pairs)} Q&A pairs from extracted chunks.")

    save_qa_pairs(qa_pairs, "generated_initial_info_qa.json")
    print("Saved parsed Q&A pairs to generated_initial_info_qa.json")

if __name__ == "__main__":
    main()
