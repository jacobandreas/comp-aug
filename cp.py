#!/usr/bin/env python3

import hlog
from seq import Encoder, SimpleDecoder, AttDecoder
from vocab import Vocab

from collections import namedtuple
import numpy as np
from scipy import stats
import torch
from torch import nn, optim
import torch.utils.data as torch_data

DEVICE=torch.device('cuda:0')

N_SYMS = 5
N_EMB = 64
N_HID = 512
N_LAYERS = 1

BATCH_SIZE = 64
TRAIN_LENS = [3, 4, 5, 6, 12, 13]
VAL_LENS = [7, 8, 9, 10, 11]


class Dataset(torch_data.Dataset):
    def __init__(self, val=False):
        self.vocab = Vocab()
        self.syms = [chr(ord('a') + i) for i in range(N_SYMS)]
        for s in self.syms:
            self.vocab.add(s)
        self.val = val

    def __len__(self):
        return 10 * BATCH_SIZE

    def __getitem__(self, i):
        if self.val:
            seq_len = np.random.choice(VAL_LENS)
        else:
            seq_len = np.random.choice(TRAIN_LENS)
        seq = [np.random.choice(self.syms) for _ in range(seq_len)]
        return self.vocab.encode(seq)

    def collate(self, seqs):
        n_batch = len(seqs)
        pad = self.vocab.pad()
        data = np.full((n_batch, max(len(s) for s in seqs)), pad, dtype=np.int64)
        for i in range(n_batch):
            data[i, :len(seqs[i])] = seqs[i]
        return [s[1:-1] for s in seqs], torch.tensor(data).to(DEVICE)

class Model(nn.Module):
    def __init__(self, dataset, n_embed, n_hidden, n_layers):
        super().__init__()
        self.encoder = Encoder(dataset.vocab, n_embed, n_hidden, n_layers,
            bidirectional=False)
        self.decoder = AttDecoder(dataset.vocab, n_embed, n_hidden, n_hidden, n_layers)

    def forward(self, data):
        context, state = self.encoder(data)
        pred, _ = self.decoder(context, data, state)
        return pred

    def decode(self, data):
        context, state = self.encoder(data)
        dec, hiddens = self.decoder.decode(context, state, DEVICE)
        return dec, hiddens

#ActInfo = namedtuple('ActInfo', 'index seq counter completion')

def analyze(hiddens):
    n_a = N_HID * N_LAYERS
    seq_data = {i_a: [] for i_a in range(n_a)}
    count_data = []
    comp_data = []

    for i in range(len(hiddens)):
        t = torch.stack(hiddens[i]).view(-1, n_a).t()
        length = t.shape[1]
        count = list(range(length))
        comp = [float(c) / length for c in count]
        count_data += count
        comp_data += comp
        for i_a in range(n_a):
            a = t[i_a, :].detach().cpu().numpy().tolist()
            a = [aa - a[0] for aa in a]
            seq_data[i_a] += a

    count_scores = {
        i_a: stats.pearsonr(seq_data[i_a], count_data)[0]
        for i_a in range(n_a)
    }

    #comp_scores = {
    #    i_a: stats.pearsonr(seq_data[i_a], comp_data)[0]
    #    for i_a in range(n_a)
    #}

    i_counter, _ = max(count_scores.items(), key=lambda x: x[1])
    print(max(count_scores.items(), key=lambda x: x[1]))
    for hseq in hiddens:
        t = torch.stack(hseq).view(-1, n_a).t()
        s = t[i_counter, :].detach().cpu().numpy().tolist()
        print(' '.join('%0.3f' % ss for ss in s))
    print()

    #print(max(comp_scores.items(), key=lambda x: x[1]))
    #print(min(count_scores.items(), key=lambda x: x[1]))
    #print(min(comp_scores.items(), key=lambda x: x[1]))

dataset = Dataset()
val_dataset = Dataset(val=True)
model = Model(dataset, N_EMB, N_HID, N_LAYERS).to(DEVICE)
loader = torch_data.DataLoader(dataset, BATCH_SIZE, collate_fn=dataset.collate)
val_loader = torch_data.DataLoader(val_dataset, BATCH_SIZE, collate_fn=dataset.collate)

obj = nn.CrossEntropyLoss(ignore_index=dataset.vocab.pad()).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=0.001)
#sched = optim.lr_scheduler.ReduceLROnPlateau(
#    opt, factor=0.5, verbose=True
#)
sched = optim.lr_scheduler.StepLR(
    opt, step_size=10, gamma=0.1
)
for i_epoch in hlog.loop('%03d', counter=range(30)):
    epoch_loss = 0
    epoch_acc = 0
    epoch_count = 0
    for seqs, data in loader:
        n_tgts = data.shape[0] * (data.shape[1] - 1)
        pred = model(data)[:, :-1, :]
        pred = pred.contiguous().view(n_tgts, len(dataset.vocab))
        tgt = data[:, 1:].contiguous().view(n_tgts)
        loss = obj(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += float(loss)

        dec, hiddens = model.decode(data)
        acc = np.mean([d == s for d, s in zip(dec, seqs)])
        epoch_acc += acc
        epoch_count += 1

    analyze(hiddens)

    val_acc = 0
    val_count = 0
    for seqs, data in val_loader:
        dec, hiddens = model.decode(data)
        acc = np.mean([d == s for d, s in zip(dec, seqs)])
        val_acc += acc
        val_count += 1

    hlog.value('loss', epoch_loss / epoch_count)
    hlog.value('acc', epoch_acc / epoch_count)
    hlog.value('vacc', val_acc / val_count)
    #sched.step(epoch_loss)
    sched.step()
