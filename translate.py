"""Training script.

Usage:
  train.py (-h | --help)
  train.py --test_src=TEST_SRC --model=MODEL [--test_tgt=TEST_TGT]

Options:
  -h --help                 Show this screen.
  --test_src=TEST_SRC       Test source file.
  --test_tgt=TEST_TGT       Test target file.
  --model=MODEL             Checkpoint prefix [default: model].

"""
import copy
import operator
from queue import PriorityQueue

from docopt import docopt
from schema import Schema, Or, Use, SchemaError

from load import *
from train import make_model
import torch.nn as nn


class BeamSearchNode(object):
    def __init__(self, prev_node, curr_input, curr_word, log_prob, length):
        self.prev_node = prev_node
        self.curr_input = curr_input
        self.curr_word = curr_word
        self.log_prob = log_prob
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.log_prob / float(self.length + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.eval() < other.eval()

    # return comparison
    def __le__(self, other):
        return self.eval() <= other.eval()

    # return comparison
    def __ne__(self, other):
        return self.eval() != other.eval()

    # return comparison
    def __gt__(self, other):
        return self.eval() > other.eval()

    # return comparison
    def __ge__(self, other):
        return self.eval() >= other.eval()


def greedy_decode(model: nn.Module, batch: int, max_len: int = 30, start_symbol: int = 1):
    batch_size = batch.src.size(0)
    memory = model.encode(batch.src, batch.src_mask)
    ys = torch.ones(batch_size, 1).fill_(start_symbol).type_as(batch.src.data)
    pred = torch.ones(batch_size, 1).fill_(
        start_symbol).type_as(batch.src.data)
    final_out = None
    for i in range(max_len - 1):
        ys = Variable(ys)
        tgt_mask = Variable(subsequent_mask(
            ys.size(1)).type_as(batch.src.data))
        out, attns = model.decode(memory, None, ys, tgt_mask)

        final_out = out if final_out is None else torch.cat(
            [final_out, out[:, -1:, :]], dim=1)
        prob = model.generator(dec_output=out, src_full=batch.src_full, dec_attns=attns,
                               enc_output=memory, dec_embeded=model.tgt_embed(ys))
        _, next_word = torch.max(prob[:, -1:, :], dim=2)
        pred = torch.cat([pred, copy.deepcopy(next_word)], dim=1)
        next_word[next_word >= len(batch.vocab.token2id)] = 3
        ys = torch.cat([ys, copy.deepcopy(next_word)], dim=1)

    return pred.cpu().numpy().tolist()[0]


def beam_decode(model: nn.Module, batch: Batch, max_len: int = 256, start_symbol: int = 1, beam: int = 5, topk: int = 1):
    batch_size = batch.src.size(0)
    memory = model.encode(batch.src, batch.src_mask)
    prediction = []
    queue_size = 0

    for idx in range(batch_size):

        ys = torch.ones(1, 1).fill_(start_symbol).type_as(batch.src.data)

        node = BeamSearchNode(None, ys, start_symbol, 0, 0)
        queue = PriorityQueue()
        queue.put((-node.eval(), node))
        queue_size += 1

        end_nodes = []
        number_required = min((topk + 1), topk - len(end_nodes))

        while True:
            if queue_size > 2000:
                break
            score, node = queue.get()
            queue_size -= 1

            if node.length >= max_len:
                break

            if node.curr_word == batch.vocab.token2id['<EOS>'] or node.length == max_len:
                end_nodes.append((score, node))
                if len(end_nodes) >= number_required:
                    break
                else:
                    continue

            ys = Variable(node.curr_input)
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(batch.src.data))
            out, attns = model.decode(memory[idx].unsqueeze(0), None, ys, tgt_mask)

            prob = model.generator(dec_output=out, src_full=batch.src_full[idx].unsqueeze(0), dec_attns=attns,
                                   enc_output=memory[idx].unsqueeze(0), dec_embeded=model.tgt_embed(ys))

            log_probs, indexes = torch.topk(prob[:, -1:, :], beam)

            nextnodes = []

            for k in range(beam):
                next_word = indexes[0, 0, k].item()
                log_prob = log_probs[0, 0, k].item()
                next_input = next_word if next_word < len(batch.vocab.token2id) else 3
                next_input = torch.cat([node.curr_input, torch.ones(1, 1).fill_(next_input).type_as(batch.src.data)],
                                       dim=1)
                node = BeamSearchNode(node, next_input, next_word, node.log_prob + log_prob, node.length + 1)

                nextnodes.append((-node.eval(), node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                queue.put((score, nn))
                queue_size += 1

        # print(len(end_nodes))
        if len(end_nodes) == 0:
            end_nodes = [queue.get() for _ in range(topk)]

        utterances = []
        for score, node in sorted(end_nodes, key=operator.itemgetter(0)):
            utterance = [node.curr_word]
            # back trace
            while node.prev_node is not None:
                node = node.prev_node
                utterance.append(node.curr_word)

            utterance = utterance[::-1]
            utterances.append(utterance)

        prediction.append(utterances[0])

    return prediction


if __name__ == "__main__":

    args = docopt(__doc__, version='0.1')

    schema = Schema({
        '--help': Or(None, Use(bool)),
        '--test_src': Use(lambda x: open(x, "r"), error='TEST_SRC should be readable'),
        '--test_tgt': Or(None, Use(lambda x: open(x, "r"), error='TEST_TGT should be readable')),
        '--model': Use(str)
    })

    try:
        args = schema.validate(args)
    except SchemaError as e:
        exit(e)

    pickled_data = torch.load(args['--model'])
    model_state, resources, parameters = pickled_data['model'], pickled_data['resources'], pickled_data['parameters']
    device = parameters['--device']

    base_vocab = resources['base_vocab']
    V = len(base_vocab)
    model = make_model(V, V, layers=parameters['--layers'],
                       d_model=parameters['--d_model'],
                       d_ff=parameters['--d_ff'],
                       h=parameters['--heads'],
                       dropout=parameters['--dropout'],
                       device=parameters['--device'])
    model.load_state_dict(model_state)
    model.eval()

    test_dataset = base_vocab.load_dataset(
        args['--test_src'], args['--test_tgt']
    )

    with torch.no_grad():

        for batch in Batch.from_dataset(test_dataset, base_vocab, batch_size=parameters['--batch_size'], device=device):
            pred = beam_decode(model, batch, start_symbol=1, max_len=256)

            print('*' * 20)
            print(' '.join(batch.vocab.id2token[i] for i in batch.src[0].cpu(
            ).data.numpy() if i not in [0, 1, 2]))
            print(' '.join(batch.ext_vocab.id2token[i] for i in batch.src_full[0].cpu(
            ).data.numpy() if i not in [0, 1, 2]))
            print(' '.join(batch.ext_vocab.id2token[i] for i in pred[0] if i not in [0, 1, 2]))
