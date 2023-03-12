import pytorch_lightning as pl

from model import RNN_VAE


def inference():
    model = RNN_VAE('E:/datasets/flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/epoch=61-step=90024.ckpt')


if __name__ == '__main__':
    inference()


