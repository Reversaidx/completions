import evaluate
from transformers import pipeline
from tqdm import tqdm

from src.data_utils import load_and_prepare_data


def evaluate_transformer_with_rouge(test_texts, max_samples=100):
    """Оценивает transformer pipeline с ROUGE-1"""
    generator = pipeline("text-generation", model="distilgpt2")
    rouge = evaluate.load("rouge")
    
    generated_texts = []
    reference_texts = []
    
    for text in tqdm(test_texts[:max_samples]):
        words = text.split()
        if len(words) < 3:
            continue
            
        prompt = " ".join(words[:2])
        reference = text.strip()
        
        try:
            result = generator(prompt, max_length=len(words) + 5, do_sample=True, 
                             temperature=0.7, pad_token_id=generator.tokenizer.eos_token_id)
            generated = result[0]["generated_text"].strip()
            
            generated_texts.append(generated)
            reference_texts.append(reference)
            
        except Exception:
            continue
    
    results = rouge.compute(predictions=generated_texts, references=reference_texts)
    return results


def main():
    _, _, test_texts = load_and_prepare_data()
    
    rouge_results = evaluate_transformer_with_rouge(test_texts)
    
    for metric, score in rouge_results.items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()