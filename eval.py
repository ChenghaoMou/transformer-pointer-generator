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


def greedy_decode(model, batch, max_len=30, start_symbol=1):
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


def beam_decode(model, batch, max_len=30, start_symbol=1, beam=2, topk=1):
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
            if node.curr_word == batch.vocab.token2id['<EOS>'] or node.length == max_len:
                end_nodes.append((score, node))
                if len(end_nodes) >= number_required:
                    break
                else:
                    continue

            ys = Variable(node.curr_input)
            tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(batch.src.data))
            out, attns = model.decode(memory[idx].unsqueeze(0), None, ys, tgt_mask)
            # final_out = out if final_out is None else torch.cat(
            #     [final_out, out[:, -1:, :]], dim=1)
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

            # print(nextnodes)

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


# def beam_decode2(model, batch, start_symbol, max_len=30, beam=5, topk=1, device='cpu'):
#
#     decoded_batch = []
#     batch_size = batch.src.size(0)
#     encoder_outputs = model.encode(batch.src, batch.src_mask)
#
#     # decoding goes sentence by sentence
#     for idx in tqdm(range(batch_size)):
#
#         encoder_output = encoder_outputs[idx].unsqueeze(0) # [1, T, H]
#         decoder_input = start_symbol
#
#         # Number of sentence to generate
#         endnodes = []
#         number_required = min((topk + 1), topk - len(endnodes))
#
#         # starting node -  decoder_attns, previous node, word id, logp, length
#         node = BeamSearchNode(None, None, decoder_input, 0, 1)
#         nodes = PriorityQueue()
#
#         # start the queue
#         nodes.put((-node.eval(), node))
#         qsize = 1
#
#         # start beam search
#         while True:
#             # give up when decoding takes too long
#             if qsize > 2000:
#                 break
#
#             # fetch the best node
#             score, n = nodes.get()
#             decoder_input = []
#             x = n
#             while x is not None:
#                 decoder_input.append(x.wordid)
#                 x = x.prevNode
#             decoder_input = decoder_input[::-1]  # [B==1, S]
#             decoder_input = Variable(torch.from_numpy(
#                 np.array(decoder_input)), requires_grad=False).long().to(device).reshape(1, -1)
#             decoder_input[decoder_input >= len(batch.vocab.token2id)] = 3
#
#             if n.wordid == batch.vocab.token2id['<EOS>'] and n.prevNode is not None:
#                 endnodes.append((score, n))
#                 # if we reached maximum # of sentences required
#                 if len(endnodes) >= number_required:
#                     break
#                 else:
#                     continue
#
#             # decode for one step using decoder
#             # tgt_mask = Variable(subsequent_mask(decoder_input.size(1)).type_as(batch.src.data))
#             # print(decoder_input.shape, tgt_mask.shape, encoder_output.shape)
#             # decoder_output, decoder_attns = model.decode(encoder_output, batch.src_mask[idx].unsqueeze(0), decoder_input, tgt_mask)
#             # print(decoder_input.shape, )
#
#             tgt_mask = Variable(subsequent_mask(decoder_input.size(1)).type_as(batch.src.data))
#             decoder_output, decoder_attns = model.decode(encoder_output, batch.src_mask[idx].unsqueeze(0), decoder_input, tgt_mask)
#
#             prob = model.generator(dec_output=decoder_output, src_full=batch.src_full[idx].unsqueeze(0), dec_attns=decoder_attns,
#                                    enc_output=encoder_output, dec_embeded=model.tgt_embed(decoder_input))
#
#             # [B, S, V]
#             log_prob, indexes = torch.topk(torch.log(prob[:, -1:, :]), beam)
#             # print(log_prob.shape, indexes.shape)
#             nextnodes = []
#
#             for new_k in range(beam):
#                 decoded_t = indexes[0, 0, new_k].item()
#                 log_p = log_prob[0, 0, new_k].item()
#
#                 node = BeamSearchNode(
#                     decoder_attns, n, int(decoded_t), n.logp + log_p, n.leng + 1)
#                 score = - node.eval()
#                 nextnodes.append((score, node))
#
#             # put them into queue
#             for i in range(len(nextnodes)):
#                 score, nn = nextnodes[i]
#                 nodes.put((score, nn))
#                 # increase qsize
#             qsize += len(nextnodes) - 1
#
#         # choose nbest paths, back trace them
#         if len(endnodes) == 0:
#             endnodes = [nodes.get() for _ in range(topk)]
#
#         utterances = []
#         for score, n in sorted(endnodes, key=operator.itemgetter(0), reverse=True):
#             utterance = []
#             utterance.append(n.wordid)
#             # back trace
#             while n.prevNode is not None:
#                 n = n.prevNode
#                 utterance.append(n.wordid)
#
#             utterance = utterance[::-1]
#             utterances.append(utterance)
#
#         decoded_batch.append(utterances)
#
#     return decoded_batch[0]
#


if __name__ == "__main__":

    args = docopt(__doc__, version='0.1')
    # print(args)

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

    # print(args)

    pickled_data = torch.load(args['--model'])
    model_state, resources, parameters = pickled_data['model'], pickled_data['resources'], pickled_data['parameters']
    device = parameters['--device']

    base_vocab = resources['base_vocab']
    V = len(base_vocab)
    model = make_model(V, V, N=parameters['--layers'],
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

    for batch in Batch.from_dataset(test_dataset, base_vocab, batch_size=parameters['--batch_size'], device=device):
        pred = beam_decode(model, batch, start_symbol=1, max_len=30)
        # print(pred)
        print('*' * 20)
        print(' '.join(batch.vocab.id2token[i] for i in batch.src[0].cpu(
        ).data.numpy() if i not in [0, 1, 2]))
        print(' '.join(batch.ext_vocab.id2token[i] for i in batch.src_full[0].cpu(
        ).data.numpy() if i not in [0, 1, 2]))
        print(' '.join(batch.ext_vocab.id2token[i] for i in pred[0] if i not in [0, 1, 2]))
