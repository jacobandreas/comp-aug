import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, vocab, n_embed, n_hidden, n_layers):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.rnn = nn.LSTM(
            n_embed,
            n_hidden,
            n_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, data):
        emb = self.embed(data)
        out, state = self.rnn(emb)
        return out, state

class SimpleDecoder(nn.Module):
    def __init__(self, vocab, n_embed, n_hidden, n_layers):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.rnn = nn.LSTM(
            n_embed,
            n_hidden,
            n_layers,
            batch_first=True,
        )
        self.predict = nn.Linear(n_hidden, len(vocab))

    def forward(self, data, state):
        emb = self.embed(data)
        out, state = self.rnn(emb, state)
        pred = self.predict(out)
        return pred
