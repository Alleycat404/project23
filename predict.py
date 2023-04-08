import pytorch_lightning as pl

from models import RNN_VAE, RNN, RNN_CNN
def inference():
    model = RNN_CNN('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/R-D/L=0.01_epoch=39-step=58080-v1.ckpt')


if __name__ == '__main__':
    inference()
