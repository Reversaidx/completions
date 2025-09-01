import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split


def clean_string(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data(data_path="./data/raw_dataset.txt", seq_len=2, max_samples=None):
    """Загружает и подготавливает данные"""
    dataArr = []
    with open(data_path, "r") as f:
        for data in f.readlines():
            dataArr.append(data)
    
    texts = [line for line in dataArr if len(line.split()) >= seq_len]
    cleaned_texts = list(map(clean_string, texts))
    
    if max_samples is not None:
        cleaned_texts = cleaned_texts[:max_samples]
    
    test_text = cleaned_texts[round(len(cleaned_texts) * 0.9):]
    remaining_texts = cleaned_texts[:round(len(cleaned_texts) * 0.9)]
    
    train_texts, val_texts = train_test_split(remaining_texts, test_size=0.1, random_state=42)
    
    return train_texts, val_texts, test_text


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




def collate_fn(batch):
    max_len = max(len(seq[0]) for seq in batch)
    batch_size=len(batch)
    padded_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq=seq[0]
        padded_batch[i, :len(seq)] = seq

    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq=seq[0]
        attention_mask[i, :len(seq)] = 1
    target=[]
    for i,seq in enumerate(batch):
        target.append(seq[1])
    target = torch.cat(target, dim=0)
    return padded_batch,target,attention_mask


def create_data_loaders(train_texts, val_texts, test_texts, config):
    """Создает data loaders"""
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    train_dataset = MaskedBertDataset(train_texts, tokenizer)
    val_dataset = MaskedBertDataset(val_texts, tokenizer)
    test_dataset = MaskedBertDataset(test_texts, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, tokenizer
