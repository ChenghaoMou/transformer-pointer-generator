"""Training script.

Usage:
  train.py (-h | --help)
  train.py --data=DATA --model=MODEL [--smooth=SMOOTH] [--layers=LAYERS] [--batch_size=BATCH_SIZE] [--steps=STEPS] [--valid_steps=VALID_STEPS] [--save_steps=SAVE_STEPS] [--device=DEVICE] [--d_model=DMODEL] [--d_ff=DFF] [--heads=HEADS] [--dropout=DROPOUT]

Options:
  -h --help                 Show this screen.
  --data=DATA               Pickled data file.
  --smooth=SMOOTH           Smooth factor for label smoothing [default: 0.1].
  --layers=LAYERS           Number of layers in encoder/decoder [default: 6].
  --batch_size=BATCH_SIZE   Number of tokens in each batch [default: 4096].
  --steps=STEPS             Training steps [default: 200000].
  --valid_steps=VALID_STEPS     Validation steps.
  --save_steps=SAVE_STEPS   Save checkpoint steps.
  --device=DEVICE           Device [default: cpu].
  --d_model=DMODEL          d_model [default: 512].
  --d_ff=DFF                d_ff [default: 2048].
  --heads=HEADS             Number of heads [default: 8].
  --dropout=DROPOUT         Dropout [default: 0.1].
  --model=MODEL             Checkpoint prefix [default: model].

"""
import os
import pickle
import time
from collections import deque

from docopt import docopt
from schema import Schema, Or, Use, SchemaError
from tqdm import tqdm as tqdm

from load import Batch
from model import *
from typing import Callable, Iterator


def make_model(src_vocab: int, tgt_vocab: int, layers: int = 6,
               d_model: int = 512, d_ff: int = 2048, h: int = 8, dropout: int = 0.1, device: str = 'cpu'):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), layers),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        CopyGenerator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device)


def run_batch(batch: Batch, model: nn.Module, loss_compute: Callable, start_symbol: int = 1, name: str = 'Train'):
    max_len = batch.tgt_full.size(1)
    memory = model.encode(batch.src, batch.src_mask)
    batch_size, seq_len = batch.src.size()
    ys = torch.ones(batch_size, 1).fill_(
        start_symbol).type_as(batch.src.data)
    final_out = torch.zeros((batch_size, 1, 512)).to(device)
    final_pred = torch.zeros((batch_size, max_len-1)).to(device)

    for i in range(max_len - 1):
        ys = Variable(ys)
        tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(batch.src.data))
        out, attns = model.decode(memory, None,
                                  ys,
                                  tgt_mask)
        final_out = torch.cat([final_out, out[:, -1:, :]], dim=1)
        prob = model.generator(dec_output=out, src_full=batch.src_full,
                               dec_attns=attns, enc_output=memory, dec_embeded=model.tgt_embed(ys))
        _, next_word = torch.max(prob[:, -1:, :], dim=2)
        final_pred[:, i] = next_word.reshape(-1)
        next_word[next_word >= len(batch.vocab.token2id)] = 3
        ys = torch.cat([ys, copy.deepcopy(next_word)], dim=1)

    loss, perplexity = loss_compute(final_out[:, 1:, :],
                        batch.tgt_full[:, 1:],
                        batch.ntokens,
                        src_full=batch.src_full,
                        dec_attns=attns,
                        enc_output=memory,
                        dec_embeded=model.tgt_embed(ys[:, 1:]))



    invalid_targets = batch.tgt_full[:, 1:].eq(0)
    accuracy = final_pred.long().eq(batch.tgt_full[:, 1:]).masked_fill_(
        invalid_targets, 0).long().sum()

    return loss, perplexity, 100 * accuracy / batch.ntokens, batch_size


def run_epoch(data_iter: Iterator, model: nn.Module, loss_compute: Callable, start_symbol: int = 1, name: str = 'Train'):
    "Standard Training and Logging Function"

    start = time.time()
    total_tokens = 0
    total_loss = 0
    total_lines = 0
    total_accuracy = 0
    total_pp = 0
    total_size = 0

    for j, batch in enumerate(data_iter):

        max_len = batch.tgt_full.size(1)
        memory = model.encode(batch.src, batch.src_mask)
        batch_size, seq_len = batch.src.size()

        ys = torch.ones(batch_size, 1).fill_(
            start_symbol).type_as(batch.src.data)
        final_out = torch.zeros((batch_size, 1, 512)).to(device)
        final_pred = torch.zeros((batch_size, max_len-1)).to(device)

        for i in range(max_len - 1):
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
            final_pred[:, i] = next_word.reshape(-1)
            next_word[next_word >= len(batch.vocab.token2id)] = 3
            ys = torch.cat([ys, copy.deepcopy(next_word)], dim=1)

        loss, perplexity = loss_compute(final_out[:, 1:, :],
                            batch.tgt_full[:, 1:],
                            batch.ntokens,
                            src_full=batch.src_full,
                            dec_attns=attns,
                            enc_output=memory,
                            dec_embeded=model.tgt_embed(ys[:, 1:]))


        invalid_targets = batch.tgt_full[:, 1:].eq(0)
        accuracy = final_pred.long().eq(batch.tgt_full[:, 1:]).masked_fill_(
            invalid_targets, 0).long().sum()

        total_size += batch_size
        total_pp += perplexity
        total_accuracy += accuracy
        total_loss += loss
        total_tokens += batch.ntokens
        total_lines += batch.src.size(0)

    return total_loss, total_pp, 100 * total_accuracy / total_tokens, total_size


if __name__ == "__main__":

    args = docopt(__doc__, version='0.1')

    schema = Schema({
        '--help': Or(None, Use(bool)),
        '--smooth': Use(float),
        '--layers': Use(int),
        '--batch_size': Use(int),
        '--steps': Use(int),
        '--valid_steps': Use(int),
        '--save_steps': Use(int),
        '--device': Use(str),
        '--d_model': Use(int),
        '--d_ff': Use(int),
        '--heads': Use(int),
        '--dropout': Use(float),
        '--data': Use(lambda x: open(x, "rb"), error='DATA should be writable'),
        '--model': Use(str)
    })

    try:
        args = schema.validate(args)
    except SchemaError as e:
        exit(e)

    # print(args)

    device = args['--device']

    resource = pickle.loads(args['--data'].read())
    base_vocab, train_dataset, valid_dataset = resource['base_vocab'], resource['train_dataset'], resource[
        'valid_dataset']

    V = len(base_vocab)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=args['--smooth'])
    model = make_model(V, V, layers=args['--layers']).to(args['--device'])
    model_opt = get_std_opt(model)
    train_loss_compute = CopyGeneratorLossCompute(model.generator, criterion, model_opt)
    valid_loss_compute = CopyGeneratorLossCompute(model.generator, criterion, None)

    step = 0

    checkpoints = deque()
    eval_loss = float('inf')
    curr_loss = float('inf')

    pbar = tqdm(range(args['--steps']))

    while step <= args['--steps']:

        torch.cuda.empty_cache()

        data_iterator = Batch.from_dataset(train_dataset, base_vocab, batch_size=args['--batch_size'], device=device)

        for j, batch in enumerate(data_iterator):

            model.train()

            loss, perplexity, accuracy, num_sents = run_batch(batch, model, train_loss_compute)
            step += 1
            batch_mem_size = batch.src.shape[0] * batch.src.shape[1] * 4 / 1024
            pbar.set_postfix_str('Batch size: {:.2f}MB Loss: {:.2f}, Perplexity: {:.2f}, Accuracy: {:.2f}%'.format(batch_mem_size,
                                                                                                                   loss,
                                                                                                                   perplexity,
                                                                                                                   accuracy))

            if step % args['--valid_steps'] == 0 and valid_dataset is not None:

                with torch.no_grad():
                    torch.cuda.empty_cache()
                    model.eval()
                    curr_loss, perplexity, accuracy, num_sents = run_epoch(
                        Batch.from_dataset(valid_dataset, base_vocab, batch_size=args['--batch_size'], device=device),
                        model,
                        valid_loss_compute, name='Eval')

                    pbar.set_description('Eval samples: {} Loss: {:.2f}, Perplexity: {:.2f}, Accuracy: {:.2f}%'.format(num_sents,
                                                                                                                    curr_loss,
                                                                                                                    perplexity,
                                                                                                                    accuracy))

            if step % args['--save_steps'] == 0 and float(curr_loss) <= float(eval_loss):
                eval_loss = curr_loss
                if len(checkpoints) == 10:
                    os.remove(checkpoints.popleft())
                checkpoints.append(f"{args['--model']}-{step}.pt")
                torch.save({
                    'model': model.state_dict(),
                    'parameters': {
                        '--smooth': args['--smooth'],
                        '--layers': args['--layers'],
                        '--batch_size': args['--batch_size'],
                        '--device': args['--device'],
                        '--d_model': args['--d_model'],
                        '--d_ff': args['--d_ff'],
                        '--heads': args['--heads'],
                        '--dropout': args['--dropout'],
                    },
                    'resources': {
                        'base_vocab': base_vocab
                    }
                }, f"{args['--model']}-{step}.pt")

            pbar.update(1)
