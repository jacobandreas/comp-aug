import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab, n_embed, n_hidden, n_layers, bidirectional=True):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.rnn = nn.LSTM(
            n_embed,
            n_hidden,
            n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, data):
        emb = self.embed(data)
        out, state = self.rnn(emb)
        return out, state

class Decoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    # TODO
    def decode(self, context, state, device):
        n_batch = state[0].shape[1]
        data = torch.tensor([[self.vocab.sos()] for _ in range(n_batch)]).to(device)
        out = [[] for _ in range(n_batch)]
        hiddens = [[] for _ in range(n_batch)]
        for t in range(20):
            pred, state = self(context, data, state)
            best = pred.squeeze(1).argmax(dim=1)
            for i in range(n_batch):
                out[i].append(int(best[i].data))
                hiddens[i].append(state[0][:, i, :])
            data = best.unsqueeze(1)
        final = []
        final_hiddens = []
        for o, h in zip(out, hiddens):
            try:
                end = o.index(self.vocab.eos())
                final.append(o[:end])
                final_hiddens.append(h[:end])
            except ValueError:
                final.append(o)
                final_hiddens.append(h)
        return final, final_hiddens

    def forward(self, context, data, state):
        n_batch, n_seq = data.shape
        preds = []
        for t in range(n_seq):
            pred, state = self.step(context, data[:, t], state)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)
        return preds, state

class SimpleDecoder(Decoder):
    def __init__(self, vocab, n_embed, n_hidden, n_layers):
        super().__init__(vocab)
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.rnn = nn.LSTM(
            n_embed,
            n_hidden,
            n_layers,
            batch_first=True
        )
        self.predict = nn.Linear(n_hidden, len(vocab))

    def step(self, context, data, state):
        emb = self.embed(data)
        out, state = self.rnn(emb.unsqueeze(1), state)
        pred = self.predict(out).squeeze(1)
        return pred, state

    #def forward(self, data, state):
    #    emb = self.embed(data)
    #    out, state = self.rnn(emb, state)
    #    pred = self.predict(out)
    #    return pred, state

class AttDecoder(Decoder):
    def __init__(self, vocab, n_embed, n_ctx, n_hidden, n_layers):
        super().__init__(vocab)
        # TODO the Luong way
        self.att_key = nn.Linear(n_ctx, n_hidden)
        self.att_val = nn.Linear(n_ctx, n_hidden)
        self.embed = nn.Embedding(len(vocab), n_embed, vocab.pad())
        self.rnn = nn.LSTM(
            n_embed + n_hidden,
            n_hidden,
            n_layers,
            batch_first=True
        )
        self.predict = nn.Linear(n_hidden, len(vocab))

    def step(self, context, data, state):
        hid, _ = state
        key = self.att_key(context)
        hid = hid.squeeze(0).unsqueeze(1).expand_as(key)
        att = F.softmax((key * hid).sum(dim=2), 1)
        pooled = (context * att.unsqueeze(2).expand_as(context)).sum(dim=1)

        val = self.att_val(pooled)
        emb = self.embed(data)
        feats = torch.cat((emb, val), dim=1)

        out, state = self.rnn(feats.unsqueeze(1), state)
        pred = self.predict(out).squeeze(1)
        return pred, state
