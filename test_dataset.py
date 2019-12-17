import logging as log
from pathlib import Path

import torch
from mosestokenizer import MosesTokenizer

from vocab import Vocab

tokr = MosesTokenizer()

RESERVED = ['<pad>', '<unk>']

PAD_IDX = 0
UNK_IDX = 1
MAX_TYPES = 10_000
BATCH_SIZE = 256
MIN_FREQ = 5


class TextDataset:

    def __init__(self, vocab: Vocab, path: Path):
        self.vocab = vocab
        log.info(f'loading data from {path}')
        # for simplicity, loading everything to memory; on large datasets this will cause OOM

        text = [line.strip().split() for line in path.open()]

        # words to index; out-of-vocab words are replaced with UNK
        xs = [[self.vocab.word2idx.get(tok, UNK_IDX) for tok in tokss]
              for tokss in text]

        self.data = xs

        log.info(f"Found {len(self.data)} records in {path}")

    def as_batches(self, batch_size, shuffle=False):  # data already shuffled
        data = self.data
        if shuffle:
            torch.random.shuffle(data)
        for i in range(0, len(data), batch_size):  # i incrememt by batch_size
            batch = data[i: i + batch_size]  # slice
            yield self.batch_as_tensors(batch)

    @staticmethod
    def batch_as_tensors(batch):

        n_ex = len(batch)
        max_len = max(len(seq) for seq in batch)
        seqs_tensor = torch.full(size=(n_ex, max_len), fill_value=PAD_IDX,
                                 dtype=torch.long)

        for i, seq in enumerate(batch):
            seqs_tensor[i, 0:len(seq)] = torch.tensor(seq)

        return seqs_tensor
