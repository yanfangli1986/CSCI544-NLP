import argparse
import logging
import logging as log
import os
import sys
import time
from pathlib import Path

import torch
import wandb
from torch import optim, nn
from tqdm import tqdm

from BiLSTM_ATT_LM import BiLSTM_ATT_LM
from LSTM_LM import LSTM_LM
from RNN_LM import RNN_LM
from data_processing import DataProcessor
from test_dataset import TextDataset, BATCH_SIZE
from utils import save_model_object
from vocab import Vocab, MAX_TYPES, MIN_FREQ


parser = argparse.ArgumentParser("nlp_rnn")
parser.add_argument('--run_id', type=int, default=0, help='running id')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')

parser.add_argument('--model', type=str, default="RNN", help='choose the model: RNN; LSTM; BiLSTM_ATT')
parser.add_argument('--emb_dim', type=int, default=30, help='embedding size')
parser.add_argument('--hid', type=int, default=60, help='hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of RNN layers')
parser.add_argument('--dropout_ratio', type=float, default=0.4, help='dropout_ratio')
parser.add_argument('--n_epochs', type=int, default=2, help='number of epoch')

args = parser.parse_args()

log_format = '%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def train(model, n_epochs, batch_size, train_data, valid_data, device=torch.device('cuda')):
    log.info(f"Moving model to {device}")
    loss_func = nn.NLLLoss(reduction='none')
    model = model.cuda()  # move model to desired device
    optimizer = optim.Adam(params=model.parameters())
    log.info(f"Device for training {device}")
    losses = []
    for epoch in range(n_epochs):
        start = time.time()
        num_toks = 0
        train_loss = 0.
        n_train_batches = 0

        model.train()  # switch to train mode
        with tqdm(train_data.as_batches(batch_size=BATCH_SIZE), leave=False) as data_bar:
            for seqs in data_bar:
                seq_loss = torch.zeros(1).cuda()
                for i in range(1, seqs.size()[1] - 1):
                    # Move input to desired device
                    cur_seqs = seqs[:, :i].cuda()   # take w0...w_(i-1) python indexing
                    cur_tars = seqs[:, i].cuda()   # predict w_i

                    log_probs = model(cur_seqs)
                    seq_loss += loss_func(log_probs, cur_tars).sum() / len(seqs)

                seq_loss /= (seqs.shape[1] - 1)  # only n-1 toks are predicted
                train_loss += seq_loss.item()
                n_train_batches += 1

                optimizer.zero_grad()  # clear grads
                seq_loss.backward()
                optimizer.step()

                pbar_msg = f'Loss:{seq_loss.item():.4f}'
                # data_bar.set_postfix_str(pbar_msg)
                wandb.log({"training_loss": seq_loss.cpu()})

                log.info(pbar_msg)

                torch.cuda.empty_cache()

                # Run validation
        with torch.no_grad():
            model.eval()  # switch to inference mode -- no grads, dropouts inactive
            val_loss = 0
            n_val_batches = 0
            for seqs in valid_data.as_batches(batch_size=batch_size, shuffle=False):
                # Move input to desired device
                seq_loss = torch.zeros(1).cuda()
                for i in range(1, seqs.size()[1] - 1):
                    # Move input to desired device
                    cur_seqs = seqs[:, :i].cuda()
                    cur_tars = seqs[:, i].cuda()

                    log_probs = model(cur_seqs)
                    seq_loss += loss_func(log_probs, cur_tars).sum() / len(seqs)
                seq_loss /= (seqs.shape[1] - 1)
                val_loss += seq_loss.item()
                n_val_batches += 1

                torch.cuda.empty_cache()

        save_model_object(model)

        avg_train_loss = train_loss / n_train_batches
        avg_val_loss = val_loss / n_val_batches
        losses.append((epoch, avg_train_loss, avg_val_loss))
        log.info(f"Epoch {epoch} complete; Losses: Train={avg_train_loss:G} Valid={avg_val_loss:G}")
        wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss})
    return losses


def main():
    wandb.init(project="nlp_course", config=args)
    #data preprocessing
    # train_path = './data/train.txt'
    # dev_path = './data/dev.txt'
    # dp = DataProcessor(train_path, dev_path)
    # dp.tokenize()

    # additional data processing
    train_file = Path('./tmp/train.txt')
    vocab_file = Path('./tmp/vocab.txt')

    if not vocab_file.exists():
        train_corpus = (line.strip() for line in train_file.open())
        vocab = Vocab.from_text(train_corpus, max_types=MAX_TYPES, min_freq=MIN_FREQ)
        vocab.save(vocab_file)
    else:
        vocab = Vocab.load(vocab_file)

    log.info(f'Vocab has {len(vocab)} types')

    train_data = TextDataset(vocab=vocab, path=train_file)
    dev_data = TextDataset(vocab=vocab, path=Path('./tmp/dev.txt'))

    # model = FNN_LM(vocab_size=len(vocab), n_class=len(vocab))
    # losses = train(model, n_epochs=5, batch_size=BATCH_SIZE, train_data=train_data,
    #                valid_data=dev_data)

    if args.model == "RNN":
        model = RNN_LM(vocab_size=len(vocab), n_class=len(vocab), emb_dim=args.emb_dim, hid=args.hid,
                       dropout_rate=args.dropout_ratio, num_layers=args.num_layers)
        losses = train(model, n_epochs=args.n_epochs, batch_size=BATCH_SIZE, train_data=train_data,
                       valid_data=dev_data)
        torch.save(model.state_dict(), wandb.run.dir)

    elif args.model == "LSTM":
        model = LSTM_LM(vocab_size=len(vocab), n_class=len(vocab), emb_dim=args.emb_dim, hid=args.hid,
                        dropout_rate=args.dropout_ratio, num_layers=args.num_layers)
        losses = train(model, n_epochs=args.n_epochs, batch_size=BATCH_SIZE, train_data=train_data,
                       valid_data=dev_data)
        torch.save(model.state_dict(), wandb.run.dir)

    elif args.model == "BiLSTM-ATT":
        model = BiLSTM_ATT_LM(vocab_size=len(vocab), n_class=len(vocab), emb_dim=args.emb_dim, hid=args.hid,
                              dropout_rate=args.dropout_ratio, num_layers=args.num_layers)
        losses = train(model, n_epochs=args.n_epochs, batch_size=BATCH_SIZE, train_data=train_data,
                       valid_data=dev_data)
        torch.save(model.state_dict(), wandb.run.dir)


def main_test():
    print("model = %s, emb_dim = %s, hid_dim = %s, dropout_rate = %s" %
          (args.model, args.emb_dim, args.hid, args.dropout_ratio))

    logging.info("model = %s, emb_dim = %s, hid_dim = %s, dropout_rate = %s" %
                 (args.model, args.emb_dim, args.hid, args.dropout_ratio))


if __name__ == "__main__":
    main()
