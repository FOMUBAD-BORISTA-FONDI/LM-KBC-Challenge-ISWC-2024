import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class RelationExtractionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        subject = item['SubjectEntity']
        objects = item['ObjectEntities']
        relation = item['Relation']
        
        combined_text = f"{subject} {relation} {' '.join(objects)}"
        
        encoding = self.tokenizer.encode_plus(
            combined_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset_path = 'data/train.jsonl'
dataset = RelationExtractionDataset(dataset_path, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)


optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
total_steps = len(dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}, Learning Rate: {scheduler.get_last_lr()[0]}')


save_path = 'new-output'
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in dataloader:
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_labels.extend(batch['labels'].numpy().flatten())
        all_predictions.extend(predictions.numpy().flatten())


accuracy = accuracy_score(all_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
error_rate = 1 - accuracy

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Error Rate: {error_rate}")







# import json
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertForTokenClassification

# # Define the dataset class
# class RelationExtractionDataset(Dataset):
#     def __init__(self, file_path, tokenizer, max_length=512):
#         self.data = []
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         with open(file_path, 'r') as file:
#             for line in file:
#                 self.data.append(json.loads(line))
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
#         subject = item['SubjectEntity']
#         objects = item['ObjectEntities']
#         relation = item['Relation']
        

#         combined_text = f"{subject} {relation} {' '.join(objects)}"
        
       
#         encoding = self.tokenizer.encode_plus(
#             combined_text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
#         input_ids = encoding['input_ids'].squeeze()
#         attention_mask = encoding['attention_mask'].squeeze()
        
      
#         labels = torch.zeros(self.max_length, dtype=torch.long)
        
#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'labels': labels
#         }


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# dataset_path = 'data/train.jsonl'


# dataset = RelationExtractionDataset(dataset_path, tokenizer)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)


# optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# loss_fn = torch.nn.CrossEntropyLoss()

# for epoch in range(3):
#     model.train()
#     for batch in dataloader:
#         optimizer.zero_grad()
#         outputs = model(
#             input_ids=batch['input_ids'],
#             attention_mask=batch['attention_mask'],
#             labels=batch['labels']
#         )
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')


# save_path = 'new-output'
# os.makedirs(save_path, exist_ok=True)
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)
