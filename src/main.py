import torch.nn as nn
# импортируем библиотеки, которые пригодятся для задачи
import torch

import torch.nn as nn

import re

import random

from datasets import load_dataset

from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import BertTokenizerFast

from tqdm import tqdm

from sklearn.model_selection import train_test_split


# функция для "чистки" текстов
def clean_string(text):
    # приведение к нижнему регистру
    text = text.lower()
    # удаление всего, кроме латинских букв, цифр и пробелов
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # удаление дублирующихся пробелов, удаление пробелов по краям
    text = re.sub(r'\s+', ' ', text).strip()

    return text

dataArr=[]
with open("../data/raw_dataset.txt","r") as f:
    for data in f.readlines():
        dataArr.append(data)


seq_len = 2


# удаляем слишком короткие тексты
texts = [line for line in dataArr if len(line.split()) >= seq_len]


# "чистим" тексты
cleaned_texts = list(map(clean_string, texts))[:1000]

test_text=cleaned_texts[round(len(cleaned_texts) - ((len(cleaned_texts) + len(cleaned_texts) * 0.10))):]

# разбиение на тренировочную и валидационную выборки
val_size = 0.10

train_texts, val_texts = train_test_split(cleaned_texts[:(len(cleaned_texts) - len(test_text))], test_size=val_size, random_state=42)

print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")


# класс датасета
class MaskedBertDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.samples = []
        for line in texts:
            token_ids = tokenizer.encode(line, add_special_tokens=True, max_length=512, truncation=True)
            self.samples.append(token_ids)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


# Загружаем BERT токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# тренировочный и валидационный датасеты
train_dataset = MaskedBertDataset(train_texts, tokenizer)
val_dataset = MaskedBertDataset(val_texts, tokenizer)
test_dataset = MaskedBertDataset(test_text, tokenizer)

def collate_fn(batch):
    # batch - это список тензоров разной длины
    # что должно быть на выходе?
    max_len = max(len(seq) for seq in batch)
    batch_size=len(batch)
    padded_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq  # копируем seq в i-ую строку

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        attention_mask[i, :len(seq)] = 1

    return padded_batch,attention_mask





# даталоадеры
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64,collate_fn=collate_fn)

class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, combine="concat"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.combine = combine


        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)


        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.rnn(emb)
        hidden_out = out[:, -1, :]
        linear_out = self.fc(out)
        return linear_out

#
#
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

#
vocab_size = tokenizer.vocab_size
hidden_dim = 128

#
#
#
model = BiRNNClassifier(vocab_size, combine="concat")
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()
#
#
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    sum_loss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch, y_batch
            x_output = model(x_batch)
            loss = criterion(x_output, y_batch)
            preds = torch.argmax(x_output, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            sum_loss += loss.item()
    return sum_loss / len(loader), correct / total
#
#
# # Основной цикл обучения
n_epochs = 3
#
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.
    for x_batch,attention_mask in tqdm(train_loader):
        optimizer.zero_grad()
        x_batch=x_batch[:,:-1]
        targets=x_batch[:,1:]
        loss = criterion(model(x_batch),targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    # train_loss /= len(train_loader)
    # val_loss, val_acc = evaluate(model, val_loader)
    # print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")