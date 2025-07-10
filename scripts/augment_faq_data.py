import json
import random

INPUT_FAQ_PATH = "college_faq.json"
OUTPUT_AUGMENTED_PATH = "college_faq_augmented.json"

# Simple synonym replacement dictionary for augmentation
SYNONYMS = {
    "admission": ["enrollment", "entry", "joining"],
    "courses": ["programs", "classes", "subjects"],
    "fee": ["tuition", "charges", "cost"],
    "deadline": ["last date", "cutoff date", "final date"],
    "apply": ["register", "submit application", "enroll"],
    "scholarship": ["financial aid", "grant", "bursary"],
    "campus": ["university grounds", "college area", "school premises"],
    "faculty": ["professors", "teachers", "instructors"],
    "eligibility": ["qualification", "criteria", "requirements"],
    "hostel": ["dormitory", "residence hall", "student housing"],
    "counseling": ["advising", "guidance", "consultation"],
    "rank": ["position", "standing", "score"],
    "branch": ["department", "field", "specialization"],
    "transfer": ["move", "shift", "change"],
    "internship": ["training", "work experience", "practical placement"],
    "reservation": ["quota", "allocation", "set aside seats"],
    "documents": ["papers", "certificates", "records"],
    "exam": ["test", "assessment", "evaluation"],
}

def synonym_replace(text):
    words = text.split()
    new_words = []
    for word in words:
        lw = word.lower().strip(",.?")
        if lw in SYNONYMS and random.random() < 0.3:
            new_word = random.choice(SYNONYMS[lw])
            # Preserve capitalization
            if word[0].isupper():
                new_word = new_word.capitalize()
            new_words.append(new_word)
        else:
            new_words.append(word)
    return " ".join(new_words)

def augment_faq(faq_data, augment_factor=2):
    augmented = []
    for entry in faq_data:
        augmented.append(entry)  # original
        for _ in range(augment_factor):
            new_question = synonym_replace(entry["question"])
            augmented.append({"question": new_question, "answer": entry["answer"]})
    return augmented

def main():
    with open(INPUT_FAQ_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    augmented_data = augment_faq(faq_data)

    with open(OUTPUT_AUGMENTED_PATH, "w", encoding="utf-8") as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)

    print(f"Augmented FAQ data saved to {OUTPUT_AUGMENTED_PATH}")

if __name__ == "__main__":
    main()
