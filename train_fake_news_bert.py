import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(data_loader, desc="Training", leave=False)

    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

def eval_model(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['real', 'fake'], labels=[0, 1])
    return accuracy, report

def main():
    print("Loading data...")
    df = pd.read_csv('processed_fake_news.csv')

    if 'label_encoded' not in df.columns:
        print("'label_encoded' column not found. Creating from 'subject' column...")
        label_map = {
            'News': 0, 'Real': 0, 'REAL': 0,
            'Fake': 1, 'Fake News': 1, 'FAKE': 1, 'Politics': 1, 'politics': 1
        }
        df['label_encoded'] = df['subject'].map(label_map)

    # Drop rows with missing labels
    df = df.dropna(subset=['label_encoded'])

    df['label_encoded'] = df['label_encoded'].astype(int)

    texts = df['clean_text'].astype(str).tolist()
    labels = df['label_encoded'].tolist()

    print(f"Total samples: {len(texts)}")
    print("Splitting dataset...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
    model = model.to(device)

    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
    val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    epochs = 3
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Train loss: {train_loss:.4f}")

        val_acc, val_report = eval_model(model, val_loader)
        print(f"Validation Accuracy: {val_acc:.4f}")
        print("Classification Report:")
        print(val_report)

    print("Saving model...")
    model.save_pretrained('./fake_news_bert_model')
    tokenizer.save_pretrained('./fake_news_bert_model')
    print("Training complete!")

if __name__ == "__main__":
    main()
