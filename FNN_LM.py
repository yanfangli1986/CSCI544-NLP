import torch
import torch.nn as nn

from test_dataset import PAD_IDX


class FNN_LM(nn.Module):

    def __init__(self, vocab_size, n_class, emb_dim=50, hid=100, dropout=0.2):
        super(FNN_LM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=PAD_IDX)
        self.linear1 = nn.Linear(emb_dim, hid)
        self.linear2 = nn.Linear(hid, n_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, seqs, log_probs=True):
        """Return log Probabilities"""
        batch_size, max_len = seqs.shape
        #     print('batch_size is {}'.format(batch_size))
        embs = self.embedding(seqs)  # embs[Batch x SeqLen x EmbDim]
        #     print('shape of embs before sum is {}'.format(embs.shape))
        embs = self.dropout(embs)
        embs = embs.sum(dim=1)  # sum over all all steps in seq
        #     print('shape of embs after sum is {}'.format(embs.shape))
        hid_activated = torch.relu(self.linear1(embs))  # Non linear
        scores = self.linear2(hid_activated)

        if log_probs:
            return torch.log_softmax(scores, dim=1)
        else:
            return torch.softmax(scores, dim=1)
