import pytorch_lightning as pl

from models import RNN_VAE, RNN


def inference():
    model = RNN_VAE('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/R-D/L=0.5_epoch=27-step=40656.ckpt')

def test():
    model = RNN_VAE('flickr')
    trainer = pl.Trainer(accelerator='auto')
    trainer.test(model, ckpt_path='ckpt/epoch=45-step=66792.ckpt')


if __name__ == '__main__':
    test()


