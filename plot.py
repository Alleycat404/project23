import pytorch_lightning as pl

from models.model1 import RNN_VAE


def inference():
    model = RNN_VAE('E:/datasets/small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/epoch=31-step=46464.ckpt')


if __name__ == '__main__':
    inference()


