import numpy as np


def dataset_statistics(dataset):

    source_lengths = list(map(len, [e.src for e in dataset.examples]))
    avg_src_len = np.mean(source_lengths)
    max_src_len = np.max(source_lengths)
    min_src_len = np.min(source_lengths)
    nn_src_percentile = np.percentile(sorted(source_lengths), 99.9)

    target_lengths = list(map(len, [e.trg for e in dataset.examples]))
    avg_tgt_len = np.mean(target_lengths)
    max_tgt_len = np.max(target_lengths)
    min_tgt_len = np.min(target_lengths)
    nn_tgt_percentile = np.percentile(sorted(target_lengths), 99.9)

    return avg_src_len, min_src_len, max_src_len, nn_src_percentile, avg_tgt_len, min_tgt_len, max_tgt_len, nn_tgt_percentile