import random
import string
import torch


def random_dataset(max_len=20, num=2000, name='random', seed=0) -> None:
    """
    Generate random dataset for copy task.

    :param max_len: Max length for each sequence. max_len//2 is the minimum length.

    :param num: Number of examples.

    :param name: File prefix.

    :param seed: Random seed.

    :return: None
    """

    random.seed(seed)

    # Only letters are in the base vocabulary
    open(f'data/{name}.vocab', 'w').write('\n'.join(list(string.ascii_letters)))

    with open(f'data/{name}.src', 'w') as src, open(f'data/{name}.tgt', 'w') as tgt:

        for x in range(num):
            # Random sequence length
            seq = random.randint(max_len // 2, max_len)
            sample = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(seq)])
            src.write(' '.join(list(sample)) + '\n')
            tgt.write(' '.join(list(sample)) + '\n')


if __name__ == "__main__":
    random_dataset(30, num=3000, name='random-train', seed=42)
    random_dataset(30, num=500, name='random-val', seed=12)
    random_dataset(30, num=500, name='random-test', seed=24)
