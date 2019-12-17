import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import PAD_IDX


# https://www.aclweb.org/anthology/P16-2034/

class BiLSTM_ATT_LM(nn.Module):
    def __init__(self, vocab_size, n_class, emb_dim=50, hid=100, num_layers=1, dropout_rate=0,
                 use_bidirectional=True):
        super(BiLSTM_ATT_LM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim, padding_idx=PAD_IDX)
        self.input_size = emb_dim
        self.hidden_size = hid
        self.num_layers = num_layers
        self.output_size = n_class

        # LSTM Layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers,
                            bidirectional=use_bidirectional, batch_first=True, dropout=dropout_rate)

        # Fully connected layer for the last output
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def attention_network(self, encoder_out, final_hidden):
        encoder_out = encoder_out.permute(1, 2, 0)
        M = torch.tanh(encoder_out)
        hidden = final_hidden.squeeze(0).unsqueeze(1)

        attn_weights = torch.bmm(hidden, M)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden = torch.bmm(encoder_out, soft_attn_weights.permute(0, 2, 1)).squeeze(2)
        new_hidden = torch.tanh(new_hidden)

        return new_hidden

    def forward(self, seqs, log_probs=True):
        batch_size, max_len = seqs.shape
        embs = self.embedding(seqs)  # embs[Batch x SeqLen x EmbDim] [N, T, D]
        embs = self.dropout(embs)

        embs = embs.view(batch_size, max_len, -1)

        # Passing the input and hidden state into the model and obtaining outputs
        output, (hidden, cell) = self.lstm(embs)

        # sum bidir outputs F+B
        fbout = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        fbout = fbout.permute(1, 0, 2)
        fbhn = (hidden[-2, :, :] + hidden[-1, :, :]).unsqueeze(0)
        # print (fbhn.shape, fbout.shape)
        attn_out = self.attention_network(fbout, fbhn)
        scores = self.fc(attn_out)

        return torch.log_softmax(scores, dim=1)
