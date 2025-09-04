import torch
from transformers import BertTokenizerFast

from src.config import load_config, get_device
from src.lstm_model import LSTMLanguageModel


def test_generation():
    """"5AB8@C5B 35=5@0F8N B5:AB0 A 703@C65==>9 <>45;LN"""
    config = load_config()
    device = get_device()


    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    model = LSTMLanguageModel(tokenizer.vocab_size, config).to(device)
    
    model.load_state_dict(torch.load("./models/model_weights.pth", map_location=device))
    model.eval()
    
    test_prompts = [
        "What is"
    ]

    
    with torch.no_grad():
        for prompt in test_prompts:
            generated = model.generate(prompt, tokenizer)
            print(f"Prompt: '{prompt}'")
            print(f"Generated: {' '.join(generated)}")
            print()


if __name__ == "__main__":
    test_generation()