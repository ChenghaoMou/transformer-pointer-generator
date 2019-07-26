import random
import string


def random_dataset(seq=20, num=2000, name='random', seed=0):
    random.seed(seed)
    open(f'data/{name}.vocab', 'w').write('\n'.join(list(string.ascii_letters)))
    with open(f'data/{name}.src', 'w') as src, open(f'data/{name}.tgt', 'w') as tgt:
        for x in range(num):
            sample = ''.join(
                [random.choice(string.ascii_letters + string.digits) for n in range(seq)])
            src.write(' '.join(list(sample)) + '\n')
            tgt.write(' '.join(list(sample)) + '\n')


if __name__ == "__main__":
    random_dataset(30, num=3000, name='train', seed=42)
    random_dataset(30, num=500, name='val', seed=12)
    random_dataset(30, num=500, name='test', seed=24)
