import pytorch_lightning as pl

from models import RNN_VAE, RNN


def inference():
    model = RNN('E:/datasets/small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/epoch=39-step=58080.ckpt')


if __name__ == '__main__':
    inference()


