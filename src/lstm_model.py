import torch
from torch import nn
from transformers import BertTokenizerFast


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)
        self.rnn = nn.LSTM(config.hidden_dim, config.hidden_dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim, vocab_size)

    def forward(self, x, attention_mask):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(x.size(0))
        final_hidden = rnn_out[batch_indices, last_indices]
        out = self.dropout(final_hidden)
        return self.fc(out)

    def generate(self, text, tokenizer, max_tokens=10):
        device = next(self.parameters()).device
        x = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        x = torch.tensor(x).unsqueeze(0).to(device)
        
        for _ in range(max_tokens):
            attention_mask = torch.ones(1, x.size(1), dtype=torch.long).to(device)
            logits = self.forward(x, attention_mask)
            next_token_id = torch.argmax(logits, dim=1)

            if next_token_id.item() == tokenizer.sep_token_id:
                break

            x = torch.cat([x, next_token_id.unsqueeze(1)], dim=1)
        
        return tokenizer.convert_ids_to_tokens(x[0].tolist())
