import os
import torch
import pytorch_lightning as pl
import copy
import math
from torch.autograd import Variable
from torch.nn import NLLLoss
from torchtext import data, datasets
from torch.nn import Embedding
from torch.nn import LogSoftmax
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.optim import Adam
from torchnlp.metrics import get_moses_multi_bleu, get_accuracy, get_token_accuracy


class Transformer(pl.LightningModule):
    r"""A transformer model. User is able to modify the attributes as needed. The architechture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples::
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab)
        >>> transformer_model = nn.Transformer(src_vocab, tgt_vocab, nhead=16, num_encoder_layers=12)
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.generator = None
        self._reset_parameters()

        self.d_model = d_model
        self.vocab_size = None
        self.nhead = nhead

        self.field = data.Field(init_token='<bos>', eos_token='<eos>')
        self.padding_idx = None
        self.loss = None

        self.src_embed = None
        self.tgt_embed = None
        self.dropout = dropout

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(T, N, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        src = self.src_embed(src)
        tgt = self.tgt_embed(tgt)

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output, attn_weights = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                            tgt_key_padding_mask=tgt_key_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
        # return output, attn_weights
        return self.generator(output)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def training_step(self, batch, batch_nb):
        # REQUIRED
        # TODO, generating masks for each batch
        batch = self.fullfil(batch)
        logits = self.forward(batch.src, batch.tgt[:-1], src_mask=batch.src_mask, tgt_mask=batch.tgt_mask,
                              memory_mask=batch.memory_mask, src_key_padding_mask=batch.src_key_padding_mask,
                              tgt_key_padding_mask=batch.tgt_key_padding_mask, memory_key_padding_mask=batch.memory_key_padding_mask)

        return {'loss': self.loss(logits.reshape(-1, logits.size(-1)), batch.tgt[1:].reshape(-1))}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        batch = self.fullfil(batch)
        with torch.no_grad():
            logits = self.forward(batch.src, batch.tgt[:-1], src_mask=batch.src_mask, tgt_mask=batch.tgt_mask,
                                  memory_mask=batch.memory_mask, src_key_padding_mask=batch.src_key_padding_mask,
                                  tgt_key_padding_mask=batch.tgt_key_padding_mask, memory_key_padding_mask=batch.memory_key_padding_mask)
        # print(logits.shape)
        return {'val_loss': self.loss(logits.reshape(-1, logits.size(-1)), batch.tgt[1:].reshape(-1))}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return [NoamOptimizer(self.parameters(), self.d_model)]

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED

        mt_train = datasets.TranslationDataset(
            path='/Users/chenghaomou/Code/Code-ProjectsPyCharm/transformer-pointer-generator/data/train', exts=('.src', '.tgt'),
            fields=(('src', self.field), ('tgt', self.field)))

        if self.vocab_size is None:
            self.field.build_vocab(mt_train, max_size=8000)
            self.vocab_size = len(self.field.vocab.stoi)
            # print(len(self.field.vocab.stoi))
            self.generator = CopyGenerator(self.d_model, self.vocab_size)
            self.padding_idx = self.field.vocab.stoi['<pad>']
            self.loss = NLLLoss(ignore_index=self.padding_idx, reduction='mean')

            self.src_embed = TransformerEmbedding(self.vocab_size, self.d_model, self.dropout)
            self.tgt_embed = TransformerEmbedding(self.vocab_size, self.d_model, self.dropout)

        train_iter = data.BucketIterator(
            dataset=mt_train, batch_size=1024,
            batch_size_fn=lambda ex, bs, sz: sz + len(ex.src),
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.tgt)))

        return [batch for batch in train_iter]

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        mt_dev = datasets.TranslationDataset(
            path='data/dev', exts=('.src', '.tgt'),
            fields=(('src', self.field), ('tgt', self.field)))

        dev_iter = data.BucketIterator(
            dataset=mt_dev, batch_size=1024,
            batch_size_fn=lambda ex, bs, sz: sz + len(ex.src),
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.tgt)))

        return [batch for batch in dev_iter]

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        mt_test = datasets.TranslationDataset(
            path='data/test', exts=('.src', '.tgt'),
            fields=(('src', self.field), ('tgt', self.field)))

        test_iter = data.BucketIterator(
            dataset=mt_test, batch_size=1024,
            batch_size_fn=lambda ex, bs, sz: sz + len(ex.src),
            sort_key=lambda x: data.interleave_keys(len(x.src), len(x.tgt)))

        return [batch for batch in test_iter]

    def fullfil(self, batch):
        batch.src_mask = None
        batch.tgt_mask = self.generate_square_subsequent_mask(batch.tgt[:-1].size(0))
        batch.memory_mask = None
        batch.src_key_padding_mask = (batch.src == self.padding_idx).transpose(0, 1)
        batch.tgt_key_padding_mask = (batch.tgt[:-1] == self.padding_idx).transpose(0, 1)
        batch.memory_key_padding_mask = None

        # print(batch.src.shape, batch.src_key_padding_mask.shape)
        # print(batch.tgt.shape, batch.tgt_key_padding_mask.shape)

        return batch


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt
        attn_weights = []

        for i in range(self.num_layers):
            output, attn_weight = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)
            attn_weights.append(attn_weight)

        if self.norm:
            output = self.norm(output)

        return output, attn_weights


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the endocder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn_weight = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weight


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class CopyGenerator(Module):

    def __init__(self, d_model, vocab_size):
        super(CopyGenerator, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.generation = Linear(d_model, vocab_size, bias=False)
        self.logits = LogSoftmax(dim=-1)

    def forward(self, output):

        return self.logits(self.generation(output))


class NoamOptimizer(Adam):

    def __init__(self, params, d_model, factor=2, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9):
        # self.optimizer = Adam(params, betas=betas, eps=eps)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.lr = 0
        self.step_num = 0
        self.factor = factor

        super(NoamOptimizer, self).__init__(params, betas=betas, eps=eps)

    def step(self, closure=None):
        self.step_num += 1
        self.lr = self.lrate()
        for group in self.param_groups:
            group['lr'] = self.lr
        super(NoamOptimizer, self).step()

    def lrate(self):
        return self.factor * self.d_model ** (-0.5) * min(self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5))


class TransformerEmbedding(Module):

    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos = PositionalEncoding(d_model, dropout)

    def forward(self, x):
        return self.pos(self.embed(x))


class PositionalEncoding(Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    # noinspection PyArgumentList
    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0)], requires_grad=False)
        return self.dropout(x)
