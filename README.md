# completions

Проект для обучения и сравнения моделей автодополнения текста.

## Установка

1. Клонировать репозиторий
2. Установить зависимости:
```bash
pip install torch transformers pydantic-settings pyyaml scikit-learn tqdm rouge-score
```

## Структура проекта

- `src/` - основной код
  - `train.py` - обучение LSTM модели
  - `eval_lstm.py` - оценка LSTM модели
  - `eval_transformer_pipeline.py` - оценка DistilGPT-2
  - `test.py` - тестирование генерации
  - `lstm_model.py` - архитектура LSTM
  - `data_utils.py` - загрузка и обработка данных
  - `config.py` - конфигурация
- `configs/model.yaml` - параметры модели
- `data/raw_dataset.txt` - данные для обучения

## Запуск

### Обучение LSTM модели
```bash
python src/train.py
```

### Оценка моделей
```bash
# LSTM модель
python src/eval_lstm.py

# DistilGPT-2
python src/eval_transformer_pipeline.py
```

### Тестирование генерации
```bash
python src/test.py
```

## Конфигурация

Настройки в `configs/model.yaml`:
- `hidden_dim` - размер скрытого слоя
- `batch_size` - размер батча
- `learning_rate` - скорость обучения
- `num_epochs` - количество эпох

## Сравнение моделей

### ROUGE метрики

**Кастомная LSTM модель:**
- ROUGE-1: 0.1780
- ROUGE-2: 0.0562  
- ROUGE-L: 0.1737
- ROUGE-Lsum: 0.1755

**DistilGPT-2:**
- ROUGE-1: 0.1386
- ROUGE-2: 0.0621
- ROUGE-L: 0.1355  
- ROUGE-Lsum: 0.1358

### Выводы

**Кастомная LSTM модель показала лучшие результаты** по большинству ключевых метрик:
- На 28% выше по ROUGE-1 (0.1780 vs 0.1386)
- На 28% выше по ROUGE-L (0.1737 vs 0.1355)
- На 29% выше по ROUGE-Lsum (0.1755 vs 0.1358)

DistilGPT-2 превосходит только по ROUGE-2 (+10%), но этот показатель менее критичен для общего качества генерации.

### Рекомендации

**Использовать кастомную LSTM модель** как основную для задач дополнения текста в данной предметной области. Модель демонстрирует:
- Лучшее понимание контекста (высокий ROUGE-L)
- Более точные предсказания отдельных слов (высокий ROUGE-1)
- Стабильное качество генерации (высокий ROUGE-Lsum)
- Добавить количество эпох для лучшего предсказания 
