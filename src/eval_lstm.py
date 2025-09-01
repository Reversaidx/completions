import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm

from src.config import load_config
from src.data_utils import load_and_prepare_data, create_data_loaders
from src.lstm_model import LSTMLanguageModel


def evaluate_with_rouge(model, tokenizer, test_texts):
    """Оценивает LSTM модель с ROUGE-1"""
    rouge = evaluate.load("rouge")
    
    generated_texts = []
    reference_texts = []
    
    for text in tqdm(test_texts):
        words = text.split()
        if len(words) < 3:
            continue
            
        prompt = " ".join(words[:2])
        reference = text.strip()
        
        try:
            generated_tokens = model.generate(prompt, tokenizer, max_tokens=len(words))
            generated = " ".join(generated_tokens).replace("[CLS]", "").replace("[SEP]", "").strip()
            
            generated_texts.append(generated)
            reference_texts.append(reference)
            
        except Exception:
            continue
    
    results = rouge.compute(predictions=generated_texts, references=reference_texts)
    return results


def main(max_samples=100):
    config = load_config()
    
    train_texts, val_texts, test_texts = load_and_prepare_data(max_samples=max_samples)
    _, _, test_loader, tokenizer = create_data_loaders(train_texts, val_texts, test_texts, config)
    
    model = LSTMLanguageModel(tokenizer.vocab_size, config)
    model.load_state_dict(torch.load("models/model_weights.pth"))
    
    rouge_results = evaluate_with_rouge(model, tokenizer, test_texts)
    
    for metric, score in rouge_results.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()