import random
from copy import deepcopy
from dataclasses import dataclass
from random import shuffle
from typing import Union, List, Tuple, Optional

import numpy as np
import torch
from torch.autograd import Variable

from model import subsequent_mask


@dataclass
class Vocab:
    token2id: dict = None
    id2token: dict = None

    def __post_init__(self) -> None:
        if self.token2id is None:
            self.token2id = {
                '<PAD>': 0,
                '<SOS>': 1,
                '<EOS>': 2,
                '<UNK>': 3,
            }

        if self.id2token is None:
            self.id2token = {i: k for k, i in self.token2id.items()}

    def load_vocab(self, *vocabs, take=lambda l: l.strip('\r\n ').split('\t')[0]) -> None:
        for vocab in vocabs:
            if vocab is None:
                continue
            for line in vocab:
                token = take(line)
                if token not in self.token2id:
                    self.token2id[token] = len(self.token2id)
                    self.id2token[self.token2id[token]] = token

    def load_dataset(self,
                     src_file,
                     tgt_file,
                     learn_vocab=False,
                     ) -> List[Tuple]:

        src = []
        tgt = []

        for line in src_file:
            src_line = line.strip('\r\n ').split()
            src.append(src_line[:])
            if learn_vocab:
                for token in src_line:
                    if token not in self.token2id:
                        self.token2id[token] = len(self.token2id)
                        self.id2token[self.token2id[token]] = token

        if tgt_file:
            for line in tgt_file:
                tgt_line = line.strip('\r\n ').split()
                tgt.append(tgt_line[:])

                if learn_vocab:
                    for token in tgt_line:
                        if token not in self.token2id:
                            self.token2id[token] = len(self.token2id)
                            self.id2token[self.token2id[token]] = token
        else:
            tgt.append([])

        return list(zip(src, tgt))

    def get(self, key, default):
        return self.token2id.get(key, default)

    def __getitem__(self, key):
        return self.token2id[key]

    def __contains__(self, key):
        return key in self.token2id

    def __len__(self):
        return len(self.token2id)

    def __setitem__(self, key, value):
        self.token2id[key] = value
        self.id2token[self.token2id[key]] = value


@dataclass
class Batch:
    src: Union[List[str], np.ndarray, List[List[int]], torch.LongTensor]
    src_full: Union[List[str], np.ndarray, List[List[int]], torch.LongTensor]
    src_mask: Optional[torch.LongTensor] = None
    tgt: Union[List[str], np.ndarray, List[List[int]], torch.LongTensor] = None
    tgt_full: Union[List[str], np.ndarray,
                    List[List[int]], torch.LongTensor] = None
    tgt_mask: Optional[torch.LongTensor] = None
    vocab: Vocab = None
    ext_vocab: Vocab = None

    def __post_init__(self):
        self.ntokens = (self.tgt != 0).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    @staticmethod
    def from_dataset(dataset, base_vocab, batch_size=4096, shuffled=True, device='cpu', max_len=256):

        # sort by src length
        dataset = sorted(dataset, key=lambda x: len(x[0]))

        # random.shuffle(dataset)

        i = 0
        curr_batch = []
        curr_max_len = 0
        while i < len(dataset):
            curr_batch.append(
                [dataset[i][0][:max_len], dataset[i][1][:max_len]])
            curr_max_len = min(
                max(len(curr_batch[-1][0]), curr_max_len), max_len)
            if curr_max_len * len(curr_batch) > batch_size:
                yield Batch.from_batch_dataset(curr_batch, curr_max_len, base_vocab, shuffled, device)
                curr_batch = []
                curr_max_len = 0

            i += 1

    @staticmethod
    def from_batch_dataset(curr_batch, curr_max_len, base_vocab, shuffled=True, device='cuda'):
        ext_vocab = deepcopy(base_vocab)
        src_tensor = np.zeros((len(curr_batch), curr_max_len + 1))
        src_full_tensor = np.zeros((len(curr_batch), curr_max_len + 1))

        tgt_max_len = max(map(lambda x: len(x[1]), curr_batch))
        tgt_tensor = None
        tgt_full_tensor = None
        if tgt_max_len > 0:
            tgt_tensor = np.zeros((len(curr_batch), tgt_max_len + 2))
            tgt_full_tensor = np.zeros((len(curr_batch), tgt_max_len + 2))
        if shuffled:
            shuffle(curr_batch)
        iterator = enumerate(curr_batch)
        for i, (src, tgt) in iterator:
            src_ids = []
            src_full_ids = []
            for token in src:
                src_ids.append(base_vocab.get(token, base_vocab['<UNK>']))
                if token not in ext_vocab:
                    ext_vocab[token] = len(ext_vocab)
                    ext_vocab.id2token[ext_vocab[token]] = token
                src_full_ids.append(ext_vocab.get(token, base_vocab['<UNK>']))
            src_ids.append(base_vocab['<EOS>'])
            src_full_ids.append(base_vocab['<EOS>'])

            src_tensor[i, :len(src_ids)] = src_ids
            src_full_tensor[i, :len(src_full_ids)] = src_full_ids

            if tgt and tgt_max_len > 0:
                tgt_ids = [base_vocab['<SOS>']]
                tgt_full_ids = [base_vocab['<SOS>']]
                for token in tgt:
                    tgt_ids.append(base_vocab.get(token, base_vocab['<UNK>']))
                    tgt_full_ids.append(ext_vocab.get(
                        token, base_vocab['<UNK>']))

                tgt_ids.append(base_vocab['<EOS>'])
                tgt_full_ids.append(base_vocab['<EOS>'])

                tgt_tensor[i, :len(tgt_ids)] = tgt_ids
                tgt_full_tensor[i, :len(tgt_full_ids)] = tgt_full_ids

        src_tensor = Variable(torch.from_numpy(
            src_tensor), requires_grad=False).long().to(device)
        src_mask = (src_tensor != 0).unsqueeze(-2).to(device)
        tgt_tensor = Variable(torch.from_numpy(
            tgt_tensor), requires_grad=False).long().to(device)
        src_full_tensor = Variable(torch.from_numpy(
            src_full_tensor), requires_grad=False).long().to(device)
        tgt_full_tensor = Variable(torch.from_numpy(
            tgt_full_tensor), requires_grad=False).long().to(device)
        tgt_mask = Batch.make_std_mask(tgt_tensor, 0)

        return Batch(src_tensor, src_full_tensor, src_mask, tgt_tensor, tgt_full_tensor, tgt_mask, base_vocab,
                     ext_vocab)
