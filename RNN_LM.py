import torch
import torch.nn as nn

from vocab import PAD_IDX


class RNN_LM(nn.Module):
    def __init__(self, vocab_size, n_class, emb_dim=50, hid=100, num_layers=1, dropout_rate=0):
        super(RNN_LM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim, padding_idx=PAD_IDX)
        self.input_size = emb_dim
        self.hidden_size = hid
        self.num_layers = num_layers
        self.output_size = n_class

        # RNN Layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity='tanh',
                          dropout=dropout_rate, batch_first=True)

        # Fully connected layer for the last output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seqs, log_probs=True):
        batch_size, max_len = seqs.shape
        embs = self.embedding(seqs)  # embs[Batch x SeqLen x EmbDim] [N, T, D]
        embs = self.dropout(embs)

        hidden = self.init_hidden(batch_size)
        embs = embs.view(batch_size, max_len, -1)

        # Passing the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(embs, hidden)

        hidden = hidden.contiguous().view(-1, self.hidden_size)
        scores = self.fc(hidden)
        if log_probs:
            return torch.log_softmax(scores, dim=1)
        else:
            return torch.softmax(scores, dim=1)

    def init_hidden(self, batch_size):
        # generates the first hidden state
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
