import random
import string


def random_dataset(seq=20, num=2000):
    open('random.vocab', 'w').write('\n'.join(list(string.ascii_letters)))
    with open('random.src', 'w') as src, open('random.tgt', 'w') as tgt:
        for x in range(num):
            sample = ''.join(
                [random.choice(string.ascii_letters + string.digits) for n in range(seq)])
            src.write(' '.join(list(sample)) + '\n')
            tgt.write(' '.join(list(sample)) + '\n')


if __name__ == "__main__":
    random_dataset(30, num=3000)
