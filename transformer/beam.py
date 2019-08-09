import operator
import torch

from queue import PriorityQueue

from tqdm import tqdm


class BeamSearchNode(object):

    def __init__(self, inp, idx, prob, length):
        self.inp = inp
        self.idx = idx
        self.prob = prob
        self.len = length

    def eval(self, alpha=1.0):
        reward = 0

        return self.prob / float(self.len - 1 + 1e-6) + alpha * reward


def beam_decode(model, memory, field, device, beam=5):

    beam_width = beam
    top_k = 1
    decoded_batch = []

    for idx in tqdm(range(memory.size(0)), disable=True):

        decoder_input = torch.LongTensor([field.vocab.stoi['<bos>']]).to(device).reshape(-1, 1)

        end_nodes = []
        number_required = min((top_k + 1), top_k - len(end_nodes))

        node = BeamSearchNode(decoder_input, field.vocab.stoi['<bos>'], 0, 1)
        nodes = PriorityQueue()

        nodes.put((-node.eval(), node))
        qsize = 1

        while True:
            if qsize >= 2000:
                break

            score, n = nodes.get()
            decoder_input = n.inp.reshape(-1, 1)

            if n.len >= 256:
                break

            if n.idx == field.vocab.stoi['<eos>'] and n.inp is not None:
                end_nodes.append((score, n))
                if len(end_nodes) >= number_required:
                    break
                else:
                    continue

            decoder_output = model.decode(decoder_input, memory[idx:idx+1])    # [T, 1, V]

            log_prob, indexes = torch.topk(decoder_output, beam_width)
            next_nodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[-1][0][new_k].item()
                log_p = log_prob[-1][0][new_k].item()

                node = BeamSearchNode(torch.cat([decoder_input, torch.LongTensor([decoded_t]).reshape(-1, 1).to(device)], dim=0),
                                      decoded_t, score + log_p, n.len + 1)
                score = - node.eval()
                next_nodes.append((score, node))

            for i in range(len(next_nodes)):
                score, next_node = next_nodes[i]
                nodes.put((score, next_node))
            qsize += len(next_nodes) - 1

        if len(end_nodes) == 0:
            end_nodes = [nodes.get() for _ in range(top_k)]

        utterances = []

        for score, n in sorted(end_nodes, key=operator.itemgetter(0)):
            utterance = n.inp.cpu().detach().numpy().reshape(-1)
            utterances.append(utterance[1:])

        decoded_batch.append(utterances)

    return decoded_batch
