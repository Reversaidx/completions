from torch import nn as nn

from src.data_utils import tokenizer
from src.lstm_model import BiRNNClassifier
from src.main import x_batch, attention_mask, y_batch

vocab_size = tokenizer.vocab_size
model = BiRNNClassifier(vocab_size)
criterion = nn.CrossEntropyLoss()
n_epochs = 3
train_loss = 0.
loss = criterion(model(x_batch,attention_mask),y_batch)
PATH = 'model_weights.pth'
