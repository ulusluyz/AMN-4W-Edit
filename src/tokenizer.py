import json

class CharTokenizer:
    def __init__(self):
        self.char2idx = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2char = {0: "<pad>", 1: "<unk>", 2: "<bos>", 3: "<eos>"}
        self.vocab_size = 4

    def fit(self, texts):
        for text in texts:
            for char in text:
                if char not in self.char2idx:
                    self.char2idx[char] = self.vocab_size
                    self.idx2char[self.vocab_size] = char
                    self.vocab_size += 1

    def encode(self, text, max_len=None):
        tokens = [self.char2idx.get(c, self.char2idx["<unk>"]) for c in text]
        tokens = [self.char2idx["<bos>"]] + tokens + [self.char2idx["<eos>"]]
        if max_len is not None:
            if len(tokens) < max_len:
                tokens += [self.char2idx["<pad>"]] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
        return tokens

    def decode(self, tokens):
        res = []
        for t in tokens:
            if t in [0, 1, 2, 3]:
                continue
            res.append(self.idx2char.get(t, ""))
        return "".join(res)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.char2idx = json.load(f)
            self.char2idx = {k: int(v) for k, v in self.char2idx.items()}
            self.idx2char = {v: k for k, v in self.char2idx.items()}
            self.vocab_size = len(self.char2idx)
