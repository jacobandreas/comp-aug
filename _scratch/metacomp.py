#!/usr/bin/env python3

from collections import namedtuple
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

VOCAB_SIZE = 2
START = 1
STOP = 2
FULL_VOCAB_SIZE = VOCAB_SIZE + 3
#TRAIN_LENS = [i for i in range(20) if i % 2 == 0]
#TEST_LENS = [i for i in range(20) if i % 2 == 1]
TRAIN_LENS = [i for i in range(5)]
TEST_LENS = [i for i in range(5, 10)]
LENS = list(TRAIN_LENS) + list(TEST_LENS)
MAX_LEN = max(LENS) + 1 + 2

N_EMBED = 64
N_HIDDEN = 256

Batch = namedtuple('Batch', 'e_seq f_seq e_obs f_obs f_tgt')

def unwrap(var):
    return var.data.cpu().numpy()

def sample_copy(n, test=False):
    lens = TEST_LENS if test else TRAIN_LENS
    data = np.random.randint(VOCAB_SIZE, size=(n, MAX_LEN)) + 3
    seq = [[] for _ in range(n)]
    obs = np.zeros((MAX_LEN, n, FULL_VOCAB_SIZE))
    tgt = np.zeros((MAX_LEN, n), dtype=np.int64)
    for i in range(n):
        seq[i].append(START)
        obs[0, i, START] = 1
        l = np.random.choice(lens)
        last = l + 2
        for j in range(1, last):
            seq[i].append(data[i, j])
            obs[j, i, data[i, j]] = 1
            tgt[j-1, i] = data[i, j]
        obs[last, i, STOP] = 1
        tgt[last-1, i] = STOP
        seq[i].append(STOP)

    obs = Variable(torch.FloatTensor(obs))
    tgt = Variable(torch.LongTensor(tgt))
    return Batch(seq, seq, obs, obs, tgt)

class Decoder(nn.Module):
    def __init__(self):
        emb = N_EMBED
        hid = N_HIDDEN
        super().__init__()
        self._embed = nn.Linear(FULL_VOCAB_SIZE, emb)
        self._rnn = nn.GRU(input_size=emb, hidden_size=hid, num_layers=1)
        self._predict = nn.Linear(hid, FULL_VOCAB_SIZE)
        self._softmax = nn.Softmax(dim=1)

    def forward(self, state, inp):
        emb = self._embed(inp)
        rep, enc = self._rnn(emb, state)
        logits = self._predict(rep)
        return enc, logits

    def decode(self, init_state, max_len, sample=False):
        n_stack, n_batch, _ = init_state.shape
        out = [[START] for _ in range(n_batch)]
        tok_inp = [START for _ in range(n_batch)]
        done = [False for _ in range(n_batch)]
        state = init_state
        for _ in range(max_len):
            hot_inp = np.zeros((1, n_batch, FULL_VOCAB_SIZE))
            for i, t in enumerate(tok_inp):
                hot_inp[0, i, t] = 1
            hot_inp = Variable(torch.FloatTensor(hot_inp))
            if init_state.is_cuda:
                hot_inp = hot_inp.cuda()
            new_state, label_logits = self(state, hot_inp)
            label_logits = label_logits.squeeze(0)
            label_probs = unwrap(self._softmax(label_logits))
            new_tok_inp = []
            for i, row in enumerate(label_probs):
                if sample:
                    tok = np.random.choice(row.size, p=row)
                else:
                    tok = row.argmax()
                new_tok_inp.append(tok)
                if not done[i]:
                    out[i].append(tok)
                done[i] = done[i] or tok == STOP
            state = new_state
            tok_inp = new_tok_inp
            if all(done):
                break
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self._embed = nn.Linear(FULL_VOCAB_SIZE, N_EMBED)
        self._rnn = nn.GRU(input_size=N_EMBED, hidden_size=N_HIDDEN, num_layers=1)

    def forward(self, obs):
        emb = self._embed(obs)
        _, enc = self._rnn(emb)
        return enc

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        enc = self.encoder(batch.e_obs)
        _, dec = self.decoder(enc, batch.f_obs)
        size_l, size_n, size_v = dec.shape
        dec = dec.view(size_l * size_n, size_v)
        tgt = batch.f_tgt.view(size_l * size_n)
        loss = self.loss(dec, tgt)
        return loss

    def decode(self, batch):
        enc = self.encoder(batch.e_obs)
        dec = self.decoder.decode(enc, MAX_LEN)
        return dec

def validate(model, batch):
    pred = model.decode(batch)
    score = 0
    count = 0
    for i, (f, pf) in enumerate(zip(batch.f_seq, pred)):
        score += int(f == pf)
        count += 1
        if i < 3:
            print(f[1:-1])
            print(pf[1:-1])
            print()
    print(1. * score / count)
    print()

def main():
    model = Model()
    opt = optim.Adam(model.parameters(), 1e-3)
    for i in range(100):
        l = 0.
        for j in range(10):
            batch = sample_copy(100)
            loss = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ll, = unwrap(loss)
            l += ll

        print(l)
        validate(model, sample_copy(100))
        validate(model, sample_copy(100, test=True))
        print()
        

main()
