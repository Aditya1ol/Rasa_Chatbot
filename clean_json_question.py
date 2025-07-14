import json

# Load your JSON file
with open('college_faq_augmented.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Clean each entry
for item in data:
    q = item.get("question", "")
    a = item.get("answer", "").strip()

    # If answer is embedded in question like '\nAnswer: Fine' or 'Answer: Fine'
    if f"Answer: {a}" in q:
        # Remove it
        item["question"] = q.split(f"Answer: {a}")[0].strip()
       

# Save cleaned data
with open('cleaned_data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Cleaned data saved to cleaned_data.json")
