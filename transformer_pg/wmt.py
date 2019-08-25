import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from loguru import logger
import argparse
from utils.sentencepiece import BPELearner


def split(path, test_size=0.1, random_state=42, shuffle=True):
    p = Path(path)
    corpus = None
    with p.open() as f:
        corpus = list(map(lambda l: l.strip('\r\n '), f.readlines()))

    train, dev = train_test_split(corpus, test_size=test_size, random_state=random_state, shuffle=shuffle)
    dev, test = train_test_split(dev, test_size=0.5, random_state=random_state, shuffle=shuffle)

    for name, dataset in zip(['train', 'dev', 'test'], [train, dev, test]):
        src, tgt = zip(*[x.split('\t', 1) for x in dataset])
        src_path = p.parent / f"{name}.src"
        tgt_path = p.parent / f"{name}.tgt"
        with src_path.open(mode='w') as src_file, tgt_path.open(mode='w') as tgt_file:
            src_file.write('\n'.join(src))
            tgt_file.write('\n'.join(tgt))

        if name == "train":
            bpe = BPELearner()
            bpe.ingest(src_path)
            bpe.ingest(tgt_path)

            bpe.save('./data/bpe')

        bpe.digest(src_path, p.parent / f"{name}.bpe.src")
        bpe.digest(tgt_path, p.parent / f"{name}.bpe.tgt")

    logger.debug(f'Datasets generated. train: {len(train)} dev: {len(dev)} test: {len(test)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Split a WMT tsv file into train, dev, test files')
    parser.add_argument('--input', '-i', type=str, help='Input tsv file', default='./data/wikititles-v1.gu-en.tsv')
    parser.add_argument('--test_size', type=float, default=0.1, help='Portion of the dev+test')

    args = parser.parse_args()
    split(args.input, args.test_size)
