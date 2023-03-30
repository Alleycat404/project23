import pytorch_lightning as pl

from models import RNN_VAE, RNN, RNN_CNN


def inference():
    model = RNN_VAE('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/R-D/L=0.5_epoch=27-step=40656.ckpt')

def te():
    model = RNN_CNN('flickr')
    trainer = pl.Trainer(accelerator='auto')
    trainer.test(model, ckpt_path='ckpt/epoch=67-step=98736.ckpt')


if __name__ == '__main__':
    te()


