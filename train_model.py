import json
import torch
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoders, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        subject = item['SubjectEntity']
        relation = item['Relation']
        object_entities = item['ObjectEntities']

        inputs = self.tokenizer(
            subject, 
            relation, 
            add_special_tokens=True, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        pad_len = self.max_length - len(object_entities)
        object_ids = [self.label_encoders['ObjectEntities'].transform([obj])[0] for obj in object_entities]
        object_ids = object_ids + [-100] * pad_len  # Pad with -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'object_ids': torch.tensor(object_ids)
        }

class EntityPredictionModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(EntityPredictionModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # Access config here

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]
        logits = self.classifier(cls_output)
        return logits

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def preprocess_data(data):
    subjects = [item['SubjectEntity'] for item in data]
    relations = [item['Relation'] for item in data]
    object_entities = [obj for item in data for obj in item['ObjectEntities']]

    label_encoders = {
        'SubjectEntity': LabelEncoder().fit(subjects),
        'Relation': LabelEncoder().fit(relations),
        'ObjectEntities': LabelEncoder().fit(object_entities)
    }

    return data, label_encoders

def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            object_ids = batch['object_ids'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1, model.classifier.out_features), object_ids.view(-1))  # Use model.classifier.out_features
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

def main():
    data_path = '/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train-imp.jsonl'
    model_name = 'bert-base-uncased'
    batch_size = 16
    learning_rate = 2e-5
    epochs = 3
    max_length = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_data(data_path)
    data, label_encoders = preprocess_data(data)

    tokenizer = BertTokenizer.from_pretrained(model_name)

    dataset = CustomDataset(data, tokenizer, label_encoders, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_labels = len(label_encoders['ObjectEntities'].classes_)
    model = EntityPredictionModel(model_name, num_labels).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_model(model, dataloader, optimizer, criterion, device, epochs)

    torch.save(model.state_dict(), 'entity_prediction_model.pth')
    print("Model training completed and saved.")

if __name__ == '__main__':
    main()



# # second try resolving with GEMINI
# import json
# import torch
# from transformers import BertTokenizer, BertModel, AdamW
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, label_encoders, max_length=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.label_encoders = label_encoders
#         self.max_length = max_length
    
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         subject = item['SubjectEntity']
#         relation = item['Relation']
#         object_entities = item['ObjectEntities']


#         inputs = self.tokenizer(
#             subject, 
#             relation, 
#             add_special_tokens=True, 
#             max_length=self.max_length, 
#             padding='max_length', 
#             truncation=True, 
#             return_tensors='pt'
#         )

#         input_ids = inputs['input_ids'].squeeze()
#         attention_mask = inputs['attention_mask'].squeeze()

#         pad_len = self.max_length - len(object_entities)
#         object_ids = [self.label_encoders['ObjectEntities'].transform([obj])[0] for obj in object_entities]
#         object_ids = object_ids + [-100] * pad_len  # Pad with -100

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'object_ids': torch.tensor(object_ids)
#         }

# # class CustomDataset(Dataset):
# #     def __init__(self, data, tokenizer, label_encoders, max_length=128):
# #         self.data = data
# #         self.tokenizer = tokenizer
# #         self.label_encoders = label_encoders
# #         self.max_length = max_length

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         item = self.data[idx]
# #         subject = item['SubjectEntity']
# #         relation = item['Relation']
# #         object_entities = item['ObjectEntities']

#         # inputs = self.tokenizer(
#         #     subject, 
#         #     relation, 
#         #     add_special_tokens=True, 
#         #     max_length=self.max_length, 
#         #     padding='max_length', 
#         #     truncation=True, 
#         #     return_tensors='pt'
#         # )

#         # input_ids = inputs['input_ids'].squeeze()
#         # attention_mask = inputs['attention_mask'].squeeze()

#         # object_ids = [self.label_encoders['ObjectEntities'].transform([obj])[0] for obj in object_entities]

# #         return {
# #             'input_ids': input_ids,
# #             'attention_mask': attention_mask,
# #             'object_ids': torch.tensor(object_ids)
# #         }

# # class EntityPredictionModel(nn.Module):
# #     def __init__(self, bert_model_name, num_labels):
# #         super(EntityPredictionModel, self).__init__()
# #         self.bert = BertModel.from_pretrained(bert_model_name)
# #         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

# #     def forward(self, input_ids, attention_mask):
# #         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
# #         cls_output = outputs[1]
# #         logits = self.classifier(cls_output)
# #         return logits

# class EntityPredictionModel(nn.Module):
#     def __init__(self, bert_model_name, num_labels):
#         super(EntityPredictionModel, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # Access config here

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs[1]
#         logits = self.classifier(cls_output)
#         return logits

# def load_data(filepath):
#     with open(filepath, 'r') as f:
#         data = [json.loads(line.strip()) for line in f]
#     return data

# def preprocess_data(data):
#     subjects = [item['SubjectEntity'] for item in data]
#     relations = [item['Relation'] for item in data]
#     object_entities = [obj for item in data for obj in item['ObjectEntities']]

#     label_encoders = {
#         'SubjectEntity': LabelEncoder().fit(subjects),
#         'Relation': LabelEncoder().fit(relations),
#         'ObjectEntities': LabelEncoder().fit(object_entities)
#     }

#     return data, label_encoders

# def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             object_ids = batch['object_ids'].to(device)

#             outputs = model(input_ids, attention_mask)
#             loss = criterion(outputs.view(-1, model.config.vocab_size), object_ids.view(-1))  # Use model.config.vocab_size
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# # def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
# #     model.train()
# #     for epoch in range(epochs):
# #         total_loss = 0
# #         for batch in dataloader:
# #             optimizer.zero_grad()
# #             input_ids = batch['input_ids'].to(device)
# #             attention_mask = batch['attention_mask'].to(device)
# #             object_ids = batch['object_ids'].to(device)

# #             outputs = model(input_ids, attention_mask)
# #             loss = criterion(outputs.view(-1, model.config.vocab_size), object_ids.view(-1))
# #             # loss = criterion(outputs, object_ids)
# #             loss.backward()
# #             optimizer.step()
# #             total_loss += loss.item()
# #         print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# def main():
#     data_path = '/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train.jsonl'
#     model_name = 'bert-base-uncased'
#     batch_size = 16
#     learning_rate = 2e-5
#     epochs = 3
#     max_length = 128

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data = load_data(data_path)
#     data, label_encoders = preprocess_data(data)

#     tokenizer = BertTokenizer.from_pretrained(model_name)

#     dataset = CustomDataset(data, tokenizer, label_encoders, max_length=max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     num_labels = len(label_encoders['ObjectEntities'].classes_)
#     model = EntityPredictionModel(model_name, num_labels).to(device)

#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     train_model(model, dataloader, optimizer, criterion, device, epochs)

#     torch.save(model.state_dict(), 'entity_prediction_model.pth')
#     print("Model training completed and saved.")

# if __name__ == '__main__':
#     main()






# # Third try with GPT
# import json
# import torch
# from transformers import BertTokenizer, BertModel, AdamW
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, label_encoders, max_length=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.label_encoders = label_encoders
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         subject = item['SubjectEntity']
#         relation = item['Relation']
#         object_entities = item['ObjectEntities']

#         inputs = self.tokenizer(
#             subject, 
#             relation, 
#             add_special_tokens=True, 
#             max_length=self.max_length, 
#             padding='max_length', 
#             truncation=True, 
#             return_tensors='pt'
#         )

#         input_ids = inputs['input_ids'].squeeze()
#         attention_mask = inputs['attention_mask'].squeeze()

#         object_ids = [self.label_encoders['ObjectEntities'].transform([obj])[0] for obj in object_entities]

#         return {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#             'object_ids': torch.tensor(object_ids)
#         }

# class EntityPredictionModel(nn.Module):
#     def __init__(self, bert_model_name, num_labels):
#         super(EntityPredictionModel, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         cls_output = outputs[1]
#         logits = self.classifier(cls_output)
#         return logits

# def load_data(filepath):
#     with open(filepath, 'r') as f:
#         data = [json.loads(line.strip()) for line in f]
#     return data

# def preprocess_data(data):
#     subjects = [item['SubjectEntity'] for item in data]
#     relations = [item['Relation'] for item in data]
#     object_entities = [obj for item in data for obj in item['ObjectEntities']]

#     label_encoders = {
#         'SubjectEntity': LabelEncoder().fit(subjects),
#         'Relation': LabelEncoder().fit(relations),
#         'ObjectEntities': LabelEncoder().fit(object_entities)
#     }

#     return data, label_encoders

# def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             object_ids = batch['object_ids'].to(device)

#             outputs = model(input_ids, attention_mask)
#             loss = criterion(outputs, object_ids)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

# def main():
#     data_path = '/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train.jsonl'
#     model_name = 'bert-base-uncased'
#     batch_size = 16
#     learning_rate = 2e-5
#     epochs = 3
#     max_length = 128

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data = load_data(data_path)
#     data, label_encoders = preprocess_data(data)

#     tokenizer = BertTokenizer.from_pretrained(model_name)

#     dataset = CustomDataset(data, tokenizer, label_encoders, max_length=max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     num_labels = len(label_encoders['ObjectEntities'].classes_)
#     model = EntityPredictionModel(model_name, num_labels).to(device)

#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     train_model(model, dataloader, optimizer, criterion, device, epochs)

#     torch.save(model.state_dict(), 'entity_prediction_model.pth')
#     print("Model training completed and saved.")

# if __name__ == '__main__':
#     main()







# #First try
# #probing Implementation

# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from transformers import BertTokenizer, BertModel, AdamW
# from sklearn.preprocessing import LabelEncoder

# # Load and preprocess data
# def load_data(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data

# def preprocess_data(data):
#     subjects = [item['SubjectEntity'] for item in data]
#     predicates = [item['Predicate'] for item in data]
#     objects = [item['ObjectEntities'] for item in data]

#     subject_encoder = LabelEncoder().fit(subjects)
#     predicate_encoder = LabelEncoder().fit(predicates)
#     object_encoder = LabelEncoder().fit([obj for sublist in objects for obj in sublist])

#     for item in data:
#         item['SubjectEntity'] = subject_encoder.transform([item['SubjectEntity']])[0]
#         item['Predicate'] = predicate_encoder.transform([item['Predicate']])[0]
#         item['ObjectEntities'] = object_encoder.transform(item['ObjectEntities'])

#     return data, {
#         'SubjectEntity': subject_encoder,
#         'Predicate': predicate_encoder,
#         'ObjectEntities': object_encoder
#     }

# # Custom Dataset class
# class CustomDataset(Dataset):
#     def __init__(self, data, tokenizer, label_encoders, max_length=128):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.label_encoders = label_encoders
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         subject = self.label_encoders['SubjectEntity'].inverse_transform([item['SubjectEntity']])[0]
#         predicate = self.label_encoders['Predicate'].inverse_transform([item['Predicate']])[0]

#         input_text = f"{subject} {predicate}"
#         inputs = self.tokenizer(input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

#         return {
#             'input_ids': inputs['input_ids'].squeeze(),
#             'attention_mask': inputs['attention_mask'].squeeze(),
#             'object_ids': torch.tensor(item['ObjectEntities'])
#         }

# # Model class with probing
# class EntityPredictionModel(nn.Module):
#     def __init__(self, bert_model_name, num_labels):
#         super(EntityPredictionModel, self).__init__()
#         self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True, output_attentions=True)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.hidden_states  # Get all hidden states
#         attentions = outputs.attentions  # Get all attention weights
#         cls_output = outputs[1]
#         logits = self.classifier(cls_output)
#         return logits, hidden_states, attentions

# # Training function with probing
# def train_model(model, dataloader, optimizer, criterion, device, epochs=3):
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             optimizer.zero_grad()
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             object_ids = batch['object_ids'].to(device)

#             outputs, hidden_states, attentions = model(input_ids, attention_mask)  # Capture hidden states and attentions
#             loss = criterion(outputs, object_ids)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

#         # Probing example: analyze the hidden states and attentions
#         # Here, just printing the shape of hidden states and attentions for demonstration
#         print(f"Hidden States Shape: {hidden_states[-1].shape}")  # Print the shape of the last layer's hidden states
#         print(f"Attentions Shape: {attentions[-1].shape}")  # Print the shape of the last layer's attentions

# # Main function
# def main():
#     data_path = 'data.jsonl'
#     model_name = 'bert-base-uncased'
#     batch_size = 16
#     learning_rate = 2e-5
#     epochs = 3
#     max_length = 128

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     data = load_data(data_path)
#     data, label_encoders = preprocess_data(data)

#     tokenizer = BertTokenizer.from_pretrained(model_name)

#     dataset = CustomDataset(data, tokenizer, label_encoders, max_length=max_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     num_labels = len(label_encoders['ObjectEntities'].classes_)
#     model = EntityPredictionModel(model_name, num_labels).to(device)

#     optimizer = AdamW(model.parameters(), lr=learning_rate)
#     criterion = nn.CrossEntropyLoss()

#     train_model(model, dataloader, optimizer, criterion, device, epochs)

#     torch.save(model.state_dict(), 'entity_prediction_model.pth')
#     print("Model training completed and saved.")

# if __name__ == '__main__':
#     main()
