import evaluate
import torch
from tqdm import tqdm

from src import config
from src.data_utils import load_and_prepare_data, create_data_loaders
from src.lstm_model import LSTMLanguageModel


def main():
    rouge = evaluate.load("rouge-1")
    train_texts, val_texts, test_texts = load_and_prepare_data()
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        train_texts, val_texts, test_texts, config
    )
    checkpoint = torch.load('checkpoint.pth')
    model = LSTMLanguageModel(tokenizer.vocab_size, config)
    optimizer = torch.optim.Adam(model.parameters())


    # Загружаем состояния модели и оптимизатора
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])




    model.eval()  # или model.train() в зависимости от задачи

    generated_summaries=[]
    references=[]
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, attention_mask in tqdm(train_loader):

        generated = model.generate("hi how are", tokenizer)
        print(f"Generated: {' '.join(generated)}")
    results = rouge.compute(predictions=generated_summaries, references=references)
    print('Metrics')
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
if __name__=='main':
    main()