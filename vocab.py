class Vocab(object):
    PAD = '<pad>'
    SOS = '<s>'
    EOS = '</s>'

    def __init__(self):
        self._contents = {}
        self._rev_contents = {}
        self.add(self.PAD)
        self.add(self.SOS)
        self.add(self.EOS)

    def add(self, sym):
        if sym not in self._contents:
            i = len(self._contents)
            self._contents[sym] = i
            self._rev_contents[i] = sym
        return self._contents[sym]

    def __getitem__(self, sym):
        return self._contents[sym]

    def __len__(self):
        return len(self._contents)

    def encode(self, seq):
        return [self.sos()] + [self[i] for i in seq] + [self.eos()]

    def decode(self, seq):
        return ''.join(self._rev_contents[i] for i in seq)

    def pad(self):
        return self._contents[self.PAD]

    def sos(self):
        return self._contents[self.SOS]

    def eos(self):
        return self._contents[self.EOS]
