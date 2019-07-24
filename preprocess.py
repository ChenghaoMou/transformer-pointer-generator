"""Preprocessing script.

Usage:
  preprocess.py (-h | --help)
  preprocess.py <train_src> <train_tgt> <save_data> [--valid_src=VALID_SRC] [--valid_tgt=VALID_TGT] [--src_vocab=SRC_VOCAB] [--tgt_vocab=TGT_VOCAB]

Options:
  -h --help                 Show this screen. 
  --valid_src=VALID_SRC     Valid src file.
  --valid_tgt=VALID_TGT     Valid tgt file.
  --src_vocab=SRC_VOCAB     Source vocab file.
  --tgt_vocab=TGT_VOCAB     Target vocab file.

"""
from docopt import docopt
from schema import Schema, And, Or, Use, SchemaError
from load import Batch, Vocab
import pickle

if __name__ == "__main__":
    args = docopt(__doc__, version='0.1')

    schema = Schema({
        '--help': Or(None, Use(bool)),
        '--src_vocab': Or(None, Use(open, error='SRC_VOCAB should be readable')),
        '--tgt_vocab': Or(None, Use(open, error='TGT_VOCAB should be readable')),
        '<train_src>': Use(open, error='train_src should be readable'),
        '<train_tgt>': Use(open, error='train_tgt should be readable'),
        '--valid_src': Or(None, Use(open, error='VALID_SRC should be readable')),
        '--valid_tgt': Or(None, Use(open, error='VALID_TGT should be readable')),
        '<save_data>': Use(lambda x: open(x, "wb"), error='save_data should be writable'),
        })
    try:
        args = schema.validate(args)
    except SchemaError as e:
        exit(e)

    base_vocab = Vocab()
    base_vocab.load_vocab(
        args['--src_vocab'], args['--tgt_vocab']
    )
    train_dataset = base_vocab.load_dataset(
      args['<train_src>'], args['<train_tgt>']
    )
    if args['--valid_src'] and args['--valid_tgt']:
      valid_dataset = base_vocab.load_dataset(
          args['--valid_src'], args['--valid_tgt']
      )
    else:
      valid_dataset = None
    
    args['<save_data>'].write(pickle.dumps(
      {
          'base_vocab': base_vocab,
          'train_dataset': train_dataset,
          'valid_dataset': valid_dataset
      },
      protocol=4
    ))
    args['<save_data>'].close()

    # resource = pickle.loads(open("temp.data", "rb").read())
    # print(resource['base_vocab'])

    
