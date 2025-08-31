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
#TODO process all texts
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
            for i in range(1, len(token_ids)):
                x = token_ids[:i]      # постепенно удлиняющиеся префиксы
                y = token_ids[i:i+1]   # следующий токен
                self.samples.append((x, y))


    def __len__(self):
            return len(self.samples)


    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)



# Загружаем BERT токенизатор
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# тренировочный и валидационный датасеты
train_dataset = MaskedBertDataset(train_texts, tokenizer)
val_dataset = MaskedBertDataset(val_texts, tokenizer)
test_dataset = MaskedBertDataset(test_text, tokenizer)

def collate_fn(batch):
    # batch - это список тензоров разной длины
    # что должно быть на выходе?
    max_len = max(len(seq[0]) for seq in batch)
    batch_size=len(batch)
    padded_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq=seq[0]
        padded_batch[i, :len(seq)] = seq  # копируем seq в i-ую строку

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq=seq[0]
        attention_mask[i, :len(seq)] = 1
    target=[]
    for i,seq in enumerate(batch):
        target.append(seq[1])
    target = torch.cat(target, dim=0)
    return padded_batch,target,attention_mask





# даталоадеры
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64,collate_fn=collate_fn)

class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)


        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)


        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, attention_mask):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        # берем последнее скрытое состояние для каждой последовательности
        last_indices = attention_mask.sum(dim=1) - 1  # индексы последних реальных токенов
        batch_indices = torch.arange(x.size(0))
        final_hidden = rnn_out[batch_indices, last_indices]  # [batch, hidden_dim]
        out = self.dropout(final_hidden)
        return self.fc(out)

    def generate(self, text, max_tokens=10):
        x=tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        x = torch.tensor(x).unsqueeze(0)
        for _ in range(max_tokens):
            attention_mask = torch.ones(1, x.size(1), dtype=torch.long)  # все 1, не 0!
            logits = self.forward(x, attention_mask)
            next_token_id = torch.argmax(logits, dim=1)  # [1] - ID следующего токена

            if next_token_id.item() == tokenizer.sep_token_id:
                break

            x = torch.cat([x, next_token_id.unsqueeze(1)], dim=1)  # добавляем токен
        return tokenizer.convert_ids_to_tokens(x[0])[0]

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
model = BiRNNClassifier(vocab_size)
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
n_epochs = 5
#
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.
    for x_batch,y_batch,attention_mask in tqdm(train_loader):
        optimizer.zero_grad()
        loss = criterion(model(x_batch,attention_mask),y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    #
    #
    # train_loss /= len(train_loader)
    # val_loss, val_acc = evaluate(model, val_loader)
    # print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Accuracy: {val_acc:.2%}")


model.eval()
with torch.no_grad():
    nya=model.generate("hi how are")
    print(nya)
# bad_cases, good_cases = [], []
# with torch.no_grad():
#     for x_batch, y_batch,attention_mask in val_loader:
#         x_batch, y_batch = x_batch, y_batch
#         logits = model(x_batch,attention_mask)
#         preds = torch.argmax(logits, dim=1)
#         for i in range(len(y_batch)):
#             input_tokens = tokenizer.convert_ids_to_tokens(x_batch[i].tolist())
#             true_tok = tokenizer.convert_ids_to_tokens([y_batch[i].item()])[0]
#             pred_tok = tokenizer.convert_ids_to_tokens([preds[i].item()])[0]
#
#             if preds[i] != y_batch[i]:
#                 bad_cases.append((input_tokens, true_tok, pred_tok))
#             else:
#                 good_cases.append((input_tokens, true_tok, pred_tok))
# random.seed(42)
# bad_cases_sampled = random.sample(bad_cases, 5)
# good_cases_sampled = random.sample(good_cases, 5)
#
# print("\nSome incorrect predictions:")
# for context, true_tok, pred_tok in bad_cases_sampled:
#     print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")
#
#
# print("\nSome correct predictions:")
# for context, true_tok, pred_tok in good_cases_sampled:
#     if true_tok == pred_tok:
#         print(f"Input: {' '.join(context)} | True: {true_tok} | Predicted: {pred_tok}")