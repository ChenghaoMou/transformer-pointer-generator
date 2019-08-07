from collections import deque

import torch
import math
import os
import argparse
import time
from itertools import tee

from torchtext import datasets, data
from tqdm import tqdm
import dill

from torchnlp.metrics import get_moses_multi_bleu
from transformer.model import Transformer, ParallelTransformer
from transformer.optim import get_std_opt
from transformer.loss import LabelSmoothing, SimpleLossCompute
from transformer.utils.statistics import dataset_statistics


def run_batch(batch, model, loss_compute, pad, curr_step):

    batch.src_mask = (batch.src == pad).to(batch.src.device)
    batch.trg_mask = (batch.trg[:-1] == pad).to(batch.src.device)
    batch.num_token = (batch.trg[1:] != pad).data.sum().item()

    trg_mask = model.generate_square_subsequent_mask(batch.trg[:-1].size(0))
    N = batch.src_mask.size(1)

    out = model.forward(
        batch.src,
        batch.trg[:-1],
        src_key_padding_mask=batch.src_mask,
        tgt_key_padding_mask=batch.trg_mask,
        tgt_mask=trg_mask.unsqueeze(1).repeat([1, N, 1]).to(batch.src.device)
    )

    loss, correct = loss_compute(out, batch.trg[1:], curr_step, batch.num_token)

    translation = torch.argmax(out, dim=-1).transpose(0, 1).cpu().detach().numpy()
    translation = [''.join(field.vocab.itos[t] for t in ex).replace('▁', ' ') for ex in translation]
    reference = [''.join(field.vocab.itos[t] for t in ex).replace('▁', ' ') for ex in
                 batch.trg[1:].transpose(0, 1).cpu().detach().numpy()]

    return loss, correct, batch.num_token, translation, reference


def run_epoch(data_iter, model, loss_compute, pad):
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_ref = []
    total_hyp = []

    for batch in data_iter:
        loss, correct, num_token, translation, reference = run_batch(batch, model, loss_compute, pad, curr_step=0)
        total_loss += loss
        total_correct += correct
        total_hyp.extend(translation)
        total_ref.extend(reference)
        total_tokens += num_token

    return total_loss, total_correct, total_tokens, total_ref, total_hyp


def main(train_iter, eval_iter, model, train_loss, dev_loss, field, steps=200000,
         eval_step=1000, report_step=50, pad=0, early_stop=6):
    start = time.time()

    curr_hyp = []
    curr_ref = []
    curr_tokens = 0
    curr_loss = 0
    curr_correct = 0
    curr_step = 0

    pbar = tqdm(range(steps))
    checkpoints = deque()
    plateau = 0
    prev_eval_score = 0.

    while curr_step <= steps:

        batch = next(iter(train_iter))
        curr_step += 1

        model.train()
        loss, correct, num_token, translation, reference = run_batch(batch, model, train_loss, pad, curr_step)
        # print(loss)
        # total_hyp.extend(translation)
        # total_ref.extend(reference)
        # total_loss += loss
        # total_correct += correct
        # total_tokens += num_token

        curr_hyp.extend(translation)
        curr_ref.extend(reference)
        curr_loss += loss
        curr_correct += correct
        curr_tokens += num_token

        if curr_step % report_step == 0:
            elapsed = time.time() - start
            pbar.update(50)
            pbar.set_postfix_str(
                f"Step: {curr_step:>7}, Size: {curr_tokens // 50:>7}/B, Loss: {curr_loss / curr_tokens:>10.2f}, "
                f"Ppl: {math.exp(curr_loss / curr_tokens) if curr_loss / curr_tokens < 300 else float('inf'):>10.2f}, "
                f"Accuracy: {100 * curr_correct / curr_tokens:.2f}%, Speed: {curr_tokens // elapsed:>7}/s, "
                f"BLEU: {get_moses_multi_bleu(curr_hyp, curr_ref, lowercase=False):.10f}")

            start = time.time()

            curr_tokens = 0
            curr_correct = 0
            curr_loss = 0
            curr_ref = []
            curr_hyp = []

        if curr_step % eval_step == 0:

            if len(checkpoints) == 10:
                os.remove(checkpoints.popleft())

            torch.save({
                'model': model.state_dict(),
                'field': field,
            },
                f'data/tfm-{curr_step}.pt',
                pickle_module=dill
            )

            checkpoints.append(f'data/tfm-{curr_step}.pt')

            with torch.no_grad():
                model.eval()
                eval_iter, curr_eval_iter = tee(eval_iter)
                eval_loss, eval_correct, eval_tokens, eval_ref, eval_hyp = run_epoch(curr_eval_iter, model, dev_loss, pad)
                eval_ppl = math.exp(eval_loss / eval_tokens) if eval_loss / eval_tokens < 300 else float('inf')
                eval_acc = 100 * eval_correct / eval_tokens
                eval_bleu = get_moses_multi_bleu(eval_hyp, eval_ref, lowercase=False)

                pbar.write(f"Eval Loss: {eval_loss / eval_tokens:>10.2f}, "
                           f"Ppl: {eval_ppl:>10.2f}, "
                           f"Accuracy: {eval_acc:.2f}%, "
                           f"BLEU: {eval_bleu:.2f} {'↑' if eval_bleu > prev_eval_score else '↓'}")

                if early_stop > 0:

                    if eval_bleu <= prev_eval_score:
                        # prev_eval_score = eval_bleu
                        plateau += 1
                        if plateau == early_stop:
                            pbar.write(f'Early stopping after {curr_step} steps')
                    else:
                        prev_eval_score = eval_bleu
                        plateau = 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Training script')

    parser.add_argument('--train_prefix', required=True, type=str,
                        help='Training file prefix')
    parser.add_argument('--train_ext', required=True, nargs=2, type=str,
                        help='Training file extensions')
    parser.add_argument('--vocab_size', '-v', required=True, type=int,
                        help='Maximum vocab size')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Dropout rate')
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help='LabelSmoothing')
    parser.add_argument('--steps', type=int, required=True,
                        help='Training step')
    parser.add_argument('--valid_prefix', type=str,
                        help='Validation file prefix')
    parser.add_argument('--valid_ext', nargs=2, type=str,
                        help='Validation file extensions')
    parser.add_argument('--valid_step', type=int, default=1000,
                        help='Validation step')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size in number of tokens')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume training from checkpoint')

    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_ids = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))

    if args.resume:
        resource = torch.load(args.resume, map_location=device)
        model_dict, field = resource['model'], resource['field']
    else:
        model_dict = None
        field = data.Field(init_token='<bos>', eos_token='<eos>')

    train = datasets.TranslationDataset(path=args.train_prefix,
                                        exts=args.train_ext,
                                        fields=list(zip(['src', 'trg'], [field, field])),
                                        filter_pred=lambda e: len(e.src) <= 256 and len(e.trg) <= 256
                                        )
    dev = datasets.TranslationDataset(path=args.valid_prefix, exts=args.valid_ext,
                                      fields=list(zip(['src', 'trg'], [field, field])))

    print("""
    Average source length: {:.2f}
    Minimum source length: {}
    Maximum source length: {}
    99.9% percentile length: {}
    
    Average target length: {:.2f}
    Minimum target length: {}
    Maximum target length: {}
    99.9% percentile length: {}
    """.format(*dataset_statistics(train)))

    if not getattr(field, 'vocab'):
        field.build_vocab(train, max_size=args.vocab_size)

    vocab_size = len(field.vocab.stoi)
    pad_index = field.vocab.stoi['<pad>']

    model = ParallelTransformer(
        module=Transformer(vocab_size, dropout=args.dropout).to(device),
        device_ids=device_ids,
        output_device=device,
        dim=1
    )
    if model_dict is not None:
        model.load_state_dict(None)

    criterion = LabelSmoothing(vocab_size, padding_idx=pad_index, smoothing=args.smoothing)
    opt = get_std_opt(model)
    train_loss = SimpleLossCompute(criterion, opt, accumulation=1)
    eval_loss = SimpleLossCompute(criterion, None)

    train_iter = data.BucketIterator(train,
                                     batch_size=args.batch_size,
                                     batch_size_fn=lambda ex, bs, sz: sz + len(ex.src),
                                     device=device,
                                     shuffle=True,
                                     repeat=True,
                                     )
    eval_iter = data.BucketIterator(dev,
                                    batch_size=args.batch_size,
                                    batch_size_fn=lambda ex, bs, sz: sz + len(ex.src),
                                    device=device,
                                    train=False)

    main(train_iter, eval_iter, model, train_loss, eval_loss, field,
         steps=args.steps, eval_step=args.valid_step, report_step=50, pad=pad_index)
