import os
from transformer_pg.utils.BPE import BPELearner
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('BPE pre-processing')
    parser.add_argument('--input', '-i', nargs='+', type=str, required=True,
                        help='Input files for training')
    parser.add_argument('--files', '-f', nargs='+', type=str, required=True,
                        help='Input files for tokenization')
    parser.add_argument('--prefix', '-p', type=str, required=True,
                        help='Model prefix')
    args = parser.parse_args()

    leaner = BPELearner()
    for f in args.input:
        leaner.ingest(f)

    leaner.save(args.prefix)

    for f in args.files:
        name, ext = os.path.splitext(f)
        leaner.digest(f, f'{name}-bpe{ext}')