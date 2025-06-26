import sys
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return nlp

def predict_on_file(model_dir, input_file, output_file):
    nlp = load_model(model_dir)
    results = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                entities = nlp(text)
                results.append({"text": text, "entities": entities})
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(results, out, ensure_ascii=False, indent=2)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict_ner.py <model_dir> <input_text_file> <output_json_file>")
        sys.exit(1)
    model_dir = sys.argv[1]  # e.g., "../models/my_xml_ner_model"
    input_file = sys.argv[2] # e.g., "data/new_texts.txt"
    output_file = sys.argv[3] # e.g., "data/ner_predictions.json"
    predict_on_file(model_dir, input_file, output_file)