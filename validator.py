import json

def validate_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
                if 'subject' not in item or 'object' not in item or 'property' not in item:
                    print(f"Missing keys in line {idx + 1}: {item}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in line {idx + 1}: {e}")

data_file = '/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train.jsonl'
validate_jsonl(data_file)
