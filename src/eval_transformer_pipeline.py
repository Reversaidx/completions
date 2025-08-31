from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
result = generator("Я собираюсь", max_length=20, do_sample=True)
print(result[0]["generated_text"])
