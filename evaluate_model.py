import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(item['SubjectEntity'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        labels = torch.tensor(item['ObjectEntitiesID'])
        return {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze(), 'labels': labels}


with open('data/val.jsonl', 'r') as f:
    eval_data = [json.loads(line) for line in f]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
eval_dataset = CustomDataset(eval_data, tokenizer, max_length=128)

model = BertForTokenClassification.from_pretrained('./new-output')

training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
)

eval_result = trainer.evaluate(eval_dataset=eval_dataset)
print(f"Evaluation result: {eval_result}")
