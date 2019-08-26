import os
import sys
from pathlib import Path
from torchtext import data
from test_tube import Experiment
from transformer_pg.model import Transformer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":

    model = Transformer(batch_size=512)
    root_dir = Path(os.path.split(os.path.dirname(sys.modules['__main__'].__file__))[0])

    exp = Experiment(save_dir=root_dir / 'output')
    trainer = Trainer(experiment=exp,
                      max_nb_epochs=10,
                      print_weights_summary=False,
                      accumulate_grad_batches=2,
                      checkpoint_callback=ModelCheckpoint(filepath=root_dir / 'output' / 'models'),
                      early_stop_callback=EarlyStopping(patience=10)
                      )

    trainer.fit(model)
