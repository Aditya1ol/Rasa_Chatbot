import json
import re

def clean_question_text(question):
    # Keep only up to the first question mark (including it)
    match = re.match(r'^(.*?\?)', question)
    if match:
        return match.group(1).strip()
    else:
        return question.strip()

def main():
    input_file = 'college_faq.json'
    output_file = 'college_faq_cleaned.json'

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        original_question = entry.get('question', '')
        cleaned_question = clean_question_text(original_question)
        entry['question'] = cleaned_question

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned questions saved to {output_file}")

if __name__ == '__main__':
    main()
