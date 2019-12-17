import logging as log
from collections import Counter
from pathlib import Path
from typing import Iterator

from mosestokenizer import MosesTokenizer

tokr = MosesTokenizer()

RESERVED = ['<pad>', '<unk>']

PAD_IDX = 0
UNK_IDX = 1
MAX_TYPES = 10_000
BATCH_SIZE = 256
MIN_FREQ = 5


class Vocab:
    """ Mapper of words <--> index """

    def __init__(self, types):
        # types is list of strings
        assert isinstance(types, list)
        assert isinstance(types[0], str)

        self.idx2word = types
        self.word2idx = {word: idx for idx, word in enumerate(types)}
        assert len(self.idx2word) == len(self.word2idx)  # One-to-One

    def __len__(self):
        return len(self.idx2word)

    def save(self, path: Path):
        log.info(f'Saving vocab to {path}')
        with path.open('w') as wr:
            for word in self.idx2word:
                wr.write(f'{word}\n')

    @staticmethod
    def load(path):
        log.info(f'loading vocab from {path}')
        types = [line.strip() for line in path.open()]
        for idx, tok in enumerate(RESERVED):  # check reserved
            assert types[idx] == tok
        return Vocab(types)

    @staticmethod
    def from_text(corpus: Iterator[str], max_types: int,
                  min_freq: int = 5):
        """
        corpus: text corpus; iterator of strings
        max_types: max size of vocabulary
        min_freq: ignore word types that have fewer ferquency than this number
        """
        log.info("building vocabulary; this might take some time")
        term_freqs = Counter(tok for line in corpus for tok in line.split())
        for r in RESERVED:
            if r in term_freqs:
                log.warning(f'Found reserved word {r} in corpus')
                del term_freqs[r]
        term_freqs = list(term_freqs.items())
        log.info(f"Found {len(term_freqs)} types; given max_types={max_types}")
        term_freqs = {(t, f) for t, f in term_freqs if f >= min_freq}
        log.info(f"Found {len(term_freqs)} after dropping freq < {min_freq} terms")
        term_freqs = sorted(term_freqs, key=lambda x: x[1], reverse=True)
        term_freqs = term_freqs[:max_types]
        types = [t for t, f in term_freqs]
        types = RESERVED + types  # prepend reserved words
        return Vocab(types)
