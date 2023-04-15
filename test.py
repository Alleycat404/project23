import os
from glob import glob

import pytorch_lightning as pl

from models import RNN_VAE, RNN, RNN_CNN, RNN_VAE_1


def inference():
    model = RNN_VAE_1('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/R-D/L=0.5_epoch=27-step=40656.ckpt')

def te(ckpt: str):
    model = RNN_VAE('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    trainer.test(model, ckpt_path=ckpt)


if __name__ == '__main__':
    models = glob('ckpt/RNN_VAE/*.ckpt')
    for model in models:
        print(os.path.split(model)[1])
        te(model)


