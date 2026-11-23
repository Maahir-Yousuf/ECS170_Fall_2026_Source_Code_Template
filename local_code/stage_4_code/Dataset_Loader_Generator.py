
import csv, os, re
from collections import Counter
import torch
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader


pad_id, unk_id, eos_id = 0,1,2
Token_re = re.compile(r"[a-z0-9']+|\.{2,}|[^\sa-z0-9']")


def tokenize(text):
    text = text.replace('’', "'").replace('‘', "'")
    return Token_re.findall(text.lower())

def detokenize(tokens):
    s = ' '.join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", s)


class Dataset_Loader_Generator(Dataset_Loader):

    def __init__(self, dName=None, dDescription=None, max_len = 64, min_freq= 1):
        super().__init__(dName, dDescription, max_len)
        self.min_freq = min_freq

        self.word_to_idx = {"<PAD>": pad_id, "<UNK>": unk_id, "<EOS>": eos_id}
        self.idx_to_word = ["<PAD>", "<UNK>", "<EOS>"]
        self.jokes = []

    def load(self):
        print("Loading Data")
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rows = list(csv.reader(f))

        self.jokes = [r[-1] for r in rows[1:] if r]
        tokenized = [tokenize(i) for i in self.jokes]

        counts = Counter(t for toks in tokenized for t in toks)

        for word, c in counts.items():
            if c >= self.min_freq:
                self.word_to_idx[word] = len(self.idx_to_word)
                self.idx_to_word.append(word)

        X,y = [],[]
        seq_len = self.max_len -1

        for toks in tokenized:
            ids = [self.word_to_idx.get(t, unk_id) for t in toks][:seq_len] + [eos_id]
            inp, tar = ids[:-1], ids[1:]

            pad = seq_len - len(inp)
            X.append(inp+ [pad_id]*pad)
            y.append(tar + [pad_id]*pad)


        self.data = {
            'train': {'X': torch.LongTensor(X), 'y': torch.LongTensor(y)},
            'word2idx': self.word_to_idx,
            'idx2word': self.idx_to_word,
            'jokes': self.jokes,
        }

        print(f"Loaded {len(self.jokes)} jokes, vocab size {len(self.idx_to_word)}, "
              f"longest joke {max(len(t) for t in tokenized)} tokens")

        return self.data