import os
import tempfile
from os.path import exists
from typing import List

import sentencepiece as sp
from dataclasses import dataclass


@dataclass
class BPELearner:

    vocab_size: int = 8000
    character_coverage: float = 0.99
    corpus: List[str] = None
    tokenizer: sp.SentencePieceProcessor = None

    def __post_init__(self):
        if self.corpus is None:
            self.corpus = []

    def ingest(self, file: str):
        assert exists(file), 'File Not Found'
        with open(file) as inp:
            self.corpus.extend(inp.readlines())

    def save(self, prefix: str):
        handle, path = tempfile.mkstemp()
        os.write(handle, ''.join(self.corpus).encode('utf-8'))
        os.close(handle)

        arg = f'--input={path} --model_type=bpe --model_prefix={prefix} --vocab_size={self.vocab_size} --character_coverage={self.character_coverage}'
        sp.SentencePieceTrainer.Train(arg)
        self.load(prefix)
        os.remove(path)

    def load(self, prefix: str):
        self.tokenizer = sp.SentencePieceProcessor()
        self.tokenizer.Load(f'{prefix}.model')

    def digest(self, inp: str, out: str):
        assert self.tokenizer is not None, 'Tokenizer Not Initialized; Call load/save First'
        assert exists(inp), 'Input File Not Found'
        with open(inp) as i, open(out, "w") as o:
            for line in i:
                o.write(' '.join(self.tokenizer.EncodeAsPieces(line.strip('\r\n '))) + '\n')
