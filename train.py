from pytorch_lightning import Trainer
from transformer_pg.model import Transformer
from test_tube import Experiment
from torchtext import data
# print(help(Transformer))

model = Transformer()
exp = Experiment(save_dir='./output')
trainer = Trainer(experiment=exp, max_nb_epochs=10)
trainer.fit(model)
