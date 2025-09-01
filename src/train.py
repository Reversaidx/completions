import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertTokenizerFast

from src.config import load_config, get_device
from src.data_utils import load_and_prepare_data, create_data_loaders
from src.lstm_model import LSTMLanguageModel


def train_model(max_samples=1000):
    config = load_config()
    device = get_device()
    print(f"Using device: {device}")
    
    # Загружаем данные
    train_texts, val_texts, test_texts = load_and_prepare_data()
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")
    
    # Создаем data loaders
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        train_texts, val_texts, test_texts, config
    )
    
    # Создаем модель
    model = LSTMLanguageModel(tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Тренировочный цикл
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch, attention_mask in tqdm(train_loader):
            x_batch, y_batch, attention_mask = x_batch.to(device), y_batch.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch, attention_mask), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f}")
    
    # Сохраняем модель
    torch.save(model.state_dict(), "../models/model_weights.pth")
    
    # Тестируем генерацию
    model.eval()
    with torch.no_grad():
        generated = model.generate("hi how are", tokenizer)
        print(f"Generated: {' '.join(generated)}")
    
    return model, tokenizer


if __name__ == "__main__":
    train_model()