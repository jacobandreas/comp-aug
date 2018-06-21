#!/usr/bin/env python3

from grammar import GrammarBuilder
import hlog
from seq import Encoder, SimpleDecoder, AttDecoder
from vocab import Vocab

import numpy as np
import torch
from torch import nn, optim
import torch.utils.data as torch_data

DEVICE=torch.device('cuda:0')

BATCH_LANGS = 50
BATCH_EXAMPLES = 5
N_NT = 5
N_T = 10
N_NT_RULES = 20
N_T_RULES = 5

N_EMB = 64
N_HID = 512
N_LAYERS = 1

START = '<'
MID = '|'
END = '>'

class Dataset(torch_data.Dataset):
    def __init__(self):
        self._gb = GrammarBuilder()
        self.vocab = Vocab()
        self.start = self.vocab.add(START)
        self.mid = self.vocab.add(MID)
        self.end = self.vocab.add(END)
        for s in self._gb.symbols(N_T):
            self.vocab.add(s)

    def __len__(self):
        return 10 * BATCH_LANGS

    def __getitem__(self, i):
        grammar = self._gb.sample(N_NT, N_T, N_NT_RULES, N_T_RULES)
        samples = list(set(grammar.sample() for _ in range(2 * BATCH_EXAMPLES)))
        samples = [
            (self.vocab.encode(e), self.vocab.encode(f))
            for e, f in samples
        ]
        if len(samples) == 1:
            samples = samples + samples
        samp_in = samples[1:BATCH_EXAMPLES+1]
        samp_out = samples[0]
        while len(samp_in) < BATCH_EXAMPLES:
            samp_in += samples[1:BATCH_EXAMPLES-len(samp_in)+1]

        in_stacked = []
        for e, f in samp_in:
            in_stacked += (
                [self.start] + list(e) + [self.mid] + list(f) + [self.end]
            )
        out_e, out_f = samp_out

        return in_stacked, out_e, out_f

    def collate(self, samples):
        n_batch = len(samples)
        pad = self.vocab.pad()
        ex, out_e, out_f = zip(*samples)

        ex_data = np.full(
            (n_batch, max(len(e) for e in ex)),
            pad,
            dtype=np.int64
        )
        out_e_data = np.full(
            (n_batch, max(len(o) for o in out_e)),
            pad,
            dtype=np.int64
        )
        out_f_data = np.full(
            (n_batch, max(len(o) for o in out_f)),
            pad,
            dtype=np.int64
        )

        for i in range(n_batch):
            ex_data[i, :len(ex[i])] = ex[i]
            out_e_data[i, :len(out_e[i])] = out_e[i]
            out_f_data[i, :len(out_f[i])] = out_f[i]

        return (
            torch.tensor(ex_data).to(DEVICE),
            torch.tensor(out_e_data).to(DEVICE),
            torch.tensor(out_f_data).to(DEVICE),
            [self.vocab.decode(o[1:-1]) for o in out_f]
        )

class Model(nn.Module):
    def __init__(self, dataset, n_embed, n_hidden, n_layers):
        super().__init__()
        self.ex_encoder = Encoder(dataset.vocab, n_embed, n_hidden, n_layers)
        self.out_encoder = Encoder(dataset.vocab, n_embed, n_hidden, n_layers)
        self.out_decoder = SimpleDecoder(dataset.vocab, n_embed, n_hidden, n_layers)

    def _encode(self, ex, out_e):
        enc_ex, state_ex = self.ex_encoder(ex)
        enc_out, state_out = self.out_encoder(out_e)

        # TODO
        state_h = (state_ex[0] + state_out[0]).sum(dim=0, keepdim=True)
        state_c = (state_ex[1] + state_out[1]).sum(dim=0, keepdim=True)

        return enc_out, (state_h, state_c)

    def forward(self, ex, out_e, out_f):
        context, state = self._encode(ex, out_e)
        dec, _ = self.out_decoder(context, out_f, state)
        return dec

    def decode(self, ex, out_e):
        context, state = self._encode(ex, out_e)
        return self.out_decoder.decode(context, state, DEVICE)

class Trainer(object):
    def __init__(self, dataset, model):
        model.to(DEVICE)
        objective = nn.CrossEntropyLoss(ignore_index=dataset.vocab.pad()).to(DEVICE)
        self.dataset = dataset
        self.model = model
        self.objective = objective

    @hlog.fn('train')
    def train(self):
        opt = optim.Adam(model.parameters(), lr=0.002)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=0.5,
            verbose=True
        )
        loader = torch_data.DataLoader(dataset, BATCH_LANGS, collate_fn=dataset.collate)
        for i_epoch in hlog.loop('epoch_%03d', counter=range(1000)):
            epoch_loss = 0
            for ex, out_e, out_f, _ in loader:
                n_tgts = BATCH_LANGS * (out_f.shape[1] - 1)
                pred = self.model.forward(ex, out_e, out_f)[:, :-1, :]
                pred = pred.contiguous().view(n_tgts, len(self.dataset.vocab))
                tgt = out_f[:, 1:].contiguous().view(n_tgts)
                loss = self.objective(pred, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += float(loss)

                #dec = self.model.decode(ex, out_e)
                #print(dec)

            hlog.value('loss', epoch_loss)
            sched.step(epoch_loss)
            
            for ex, out_e, out_f, pp_f in loader:
                dec = self.model.decode(ex, out_e)
                dec = [self.dataset.vocab.decode(d) for d in dec]
                for d, p in list(zip(dec, pp_f))[:5]:
                    hlog.value('ex', '%s %s' % (d, p))
                break

dataset = Dataset()
model = Model(dataset, N_EMB, N_HID, N_LAYERS)
trainer = Trainer(dataset, model)
trainer.train()
