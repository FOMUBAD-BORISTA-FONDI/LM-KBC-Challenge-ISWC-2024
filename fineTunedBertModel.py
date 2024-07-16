import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Custom dataset class
class RelationExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = f"{item['SubjectEntity']} [SEP] {item['Relation']}"
        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = item['ObjectEntitiesID']
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def train_model(train_data, val_data, model, tokenizer, epochs=3, batch_size=16, lr=2e-5):
    train_dataset = RelationExtractionDataset(train_data, tokenizer)
    val_dataset = RelationExtractionDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")

        evaluate_model(model, val_loader)

def evaluate_model(model, val_loader):
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    nb_eval_steps = 0

    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

            loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), labels.view(-1))
            total_eval_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            total_eval_accuracy += (preds == labels).cpu().numpy().mean()

        nb_eval_steps += 1

    avg_val_accuracy = total_eval_accuracy / nb_eval_steps
    avg_val_loss = total_eval_loss / nb_eval_steps

    print(f"Validation Loss: {avg_val_loss}")
    print(f"Validation Accuracy: {avg_val_accuracy}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    model.to(device)

    data_path = "/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train.jsonl"
    data = load_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.1)

    train_model(train_data, val_data, model, tokenizer)







# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# import json
# from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# # Définir la classe de ton jeu de données personnalisé
# class MonDataset(Dataset):
#     def __init__(self, data_file, tokenizer, max_length=128):
#         self.data = self.load_data(data_file)
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         sujet = item['subject']
#         objet = item['object']
#         propriete = item['property']

#         inputs = self.tokenizer.encode_plus(
#             sujet,
#             objet,
#             add_special_tokens=True,
#             max_length=self.max_length,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt',
#             return_attention_mask=True  # Ajout pour retourner attention_mask
#         )

#         return {
#             'input_ids': inputs['input_ids'].squeeze(),  # Squeeze pour enlever la dimension batch_size 1
#             'attention_mask': inputs['attention_mask'].squeeze(),
#             'labels': torch.tensor(propriete, dtype=torch.long)
#         }

#     def load_data(self, data_file):
#         with open(data_file, 'r', encoding='utf-8') as f:
#             data = [json.loads(line) for line in f]
#         return data

# # Fonction pour entraîner le modèle
# def train_model(train_loader, model, optimizer, num_epochs=3, eval_loader=None):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     model.train()

#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)
#         print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

#         # Évaluation
#         if eval_loader is not None:
#             model.eval()
#             eval_loss, eval_accuracy, eval_f1, eval_recall = evaluate_model(model, eval_loader, device)
#             print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}, F1 Score: {eval_f1:.4f}, Recall: {eval_recall:.4f}')
#             model.train()

#     # Fin de l'entraînement
#     print("Training finished.")

# # Fonction pour évaluer le modèle
# def evaluate_model(model, eval_loader, device):
#     total_loss = 0
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in eval_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             logits = outputs.logits

#             total_loss += loss.item()

#             preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels.detach().cpu().numpy())

#     eval_loss = total_loss / len(eval_loader)
#     eval_accuracy = accuracy_score(all_labels, all_preds)
#     eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

#     return eval_loss, eval_accuracy, eval_f1, eval_recall

# # Fonction principale pour l'entraînement et la sauvegarde du modèle
# def main():
#     # Paramètres
#     data_file = '/home/borista/Desktop/New/LM-KBC-Challenge-ISWC-2024/data/train.jsonl'  # Remplace avec ton fichier JSONL
#     model_name = 'bert-base-uncased'
#     output_model_file = 'model_bert.pth'

#     # Charger le tokenizer et le modèle
#     tokenizer = BertTokenizer.from_pretrained(model_name)
#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

#     # Charger les données
#     dataset = MonDataset(data_file, tokenizer)
#     train_data, _ = train_test_split(dataset.data, test_size=0.1, random_state=42)
#     train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=torch.utils.data._utils.collate.default_collate)

#     # Optimizer
#     optimizer = AdamW(model.parameters(), lr=2e-5)

#     # Entraîner le modèle
#     train_model(train_loader, model, optimizer)

#     # Sauvegarder le modèle
#     torch.save(model.state_dict(), output_model_file)
#     print(f"Modèle sauvegardé sous {output_model_file}")

# if __name__ == "__main__":
#     main()