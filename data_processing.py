import logging as log
from pathlib import Path

from mosestokenizer import MosesTokenizer
from tqdm import tqdm

log.basicConfig(level=log.INFO)
tokr = MosesTokenizer()


class DataProcessor:
    def __init__(self, train_path, dev_path):
        self.train_path = train_path
        self.dev_path = dev_path

    def read_tokenized(self, dir):
        """Tokenization wrapper"""
        inputfile = open(dir)
        for sent in inputfile:
            yield tokr(sent.strip())

    def tokenize(self):

        train_file = Path("./tmp/" + "train.txt")
        with train_file.open('w') as w:
            for toks in tqdm(self.read_tokenized(Path(self.train_path))):
                toks.insert(0, '<bos>')
                toks.insert(len(toks), '<eos>')
                # convert numeric elements to <num> and other elements to lowercase
                toks = ['<num>' if x.isnumeric() else x.lower() for x in toks]
                # add <bos> and <eos> to the beginning and end of a sentence respectively
                w.write(" ".join(toks) + '\n')

        dev_file = Path("./tmp/" + "dev.txt")
        with dev_file.open('w') as w:
            for toks in tqdm(self.read_tokenized(Path(self.dev_path))):
                toks.insert(0, '<bos>')
                toks.insert(len(toks), '<eos>')
                # convert numeric elements to <num> and other elements to lowercase
                toks = ['<num>' if x.isnumeric() else x.lower() for x in toks]
                # add <bos> and <eos> to the beginning and end of a sentence respectively
                w.write(" ".join(toks) + '\n')
