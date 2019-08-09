import torch
import os
import argparse
from tqdm import tqdm
from transformer.beam import beam_decode
from transformer.model import Transformer, ParallelTransformer
from torchtext import data, datasets


def run_epoch(data_iter, model, field, device):

    result = []
    for i, batch in tqdm(enumerate(data_iter)):

        batch.src_mask = (batch.src == field.vocab.stoi['<pad>']).to(batch.src.device)
        mem = model.encode(batch.src, src_mask=None, src_key_padding_mask=batch.src_mask)
        mem = mem.transpose(0, 1)
        result.extend(beam_decode(model, mem, field, device, beam=5))

        print()
        print('>>' + ''.join(field.vocab.itos[x] for x in batch.src[:, -1] if x > 3).replace('▁', ' '))
        print('>>' + ''.join(field.vocab.itos[x] for x in batch.trg[:, -1] if x > 3).replace('▁', ' '))
        print('>>' + ''.join(field.vocab.itos[x] for x in result[-1][0] if x > 3).replace('▁', ' '))
        print()

    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Translate')
    parser.add_argument('--model', '-m', type=str, required=True, help='Model file')
    parser.add_argument('--field', '-f', type=str, required=True, help='Field file')
    parser.add_argument('--prefix', type=str, required=True, help='Input prefix')
    parser.add_argument('--exts', nargs=2, help='Extension')

    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_ids = [f'cuda:{i}' for i in reversed(range(torch.cuda.device_count()))]

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(torch.cuda.device_count())))

    model_dict = torch.load(args.model, map_location=device)
    field = torch.load(args.field)

    vocab_size = len(field.vocab.stoi)
    pad_index = field.vocab.stoi['<pad>']

    model = ParallelTransformer(
        module=Transformer(vocab_size).to(device),
        device_ids=device_ids,
        output_device=device,
        dim=1
    )

    model.load_state_dict(model_dict)

    test = datasets.TranslationDataset(
        path=args.prefix,
        exts=args.exts,
        fields=(('src', field), ('trg', field))
    )

    test_iter = data.BucketIterator(
        test,
        batch_size=8,
        batch_size_fn=lambda ex, bs, sz: sz + len(ex.src), device=device,
        train=False
    )

    result = run_epoch(test_iter, model, field, device)
