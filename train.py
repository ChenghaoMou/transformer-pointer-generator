import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from torch.autograd import Variable
from model import *
from tqdm import tqdm as tqdm
from load import Batch, Vocab
import pickle



def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1,
               device='cpu'):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        CopyGenerator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)


def run_epoch(data_iter, model, loss_compute, start_symbol=1, name='Train'):
    "Standard Training and Logging Function"

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    pbar = tqdm(enumerate(data_iter))

    for j, batch in pbar:
        max_len = batch.tgt_full.size(1)
        memory = model.encode(batch.src, batch.src_mask)
        batch_size, seq_len = batch.src.size()
        ys = torch.ones(batch_size, 1).fill_(
            start_symbol).type_as(batch.src.data)
        final_out = torch.zeros((batch_size, 1, 512)).to(device)

        for i in range(max_len-1):

            ys = Variable(ys)
            tgt_mask = Variable(subsequent_mask(
                ys.size(1)).type_as(batch.src.data))
            out, attns = model.decode(memory, None,
                                      ys,
                                      tgt_mask)
            final_out = torch.cat([final_out, out[:, -1:, :]], dim=1)
            prob = model.generator(dec_output=out, src_full=batch.src_full,
                                   dec_attns=attns, enc_output=memory, dec_embeded=model.tgt_embed(ys))
            _, next_word = torch.max(prob[:, -1:, :], dim=2)
            next_word[next_word >= len(batch.vocab.token2id)] = 3
            ys = torch.cat([ys, copy.deepcopy(next_word)], dim=1)

        loss = loss_compute(final_out[:, 1:, :],
                            batch.tgt_full[:, 1:],
                            batch.ntokens,
                            src_full=batch.src_full,
                            dec_attns=attns,
                            enc_output=memory,
                            dec_embeded=model.tgt_embed(ys[:, 1:]))
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if j % 50 == 1:
            
            elapsed = time.time() - start
            pbar.set_description(
                f"{name} Epoch Step: {j:>5} Loss: {loss / batch.ntokens:.2f} Tokens per Sec: {tokens / elapsed:>10}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens







if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    resource = pickle.loads(open("random.data", "rb").read())
    base_vocab, train_dataset, valid_dataset = resource['base_vocab'], resource['train_dataset'], resource['valid_dataset']
    
    V = len(base_vocab)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    model = make_model(V, V, N=2).to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400, torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(2):
        model.train()
        run_epoch(Batch.from_dataset(train_dataset, base_vocab, batch_size=500, device=device), model,
                CopyGeneratorLossCompute(model.generator, criterion, model_opt))
        model.eval()
        run_epoch(Batch.from_dataset(valid_dataset, base_vocab, batch_size=500, device=device), model,
                CopyGeneratorLossCompute(model.generator, criterion, None), name='Eval')

    torch.save(model.state_dict(), "copy-model.pt")
