import pytorch_lightning as pl
from models import RNN_VAE, RNN, RNN_CNN, RNN_VAE_1, RNN_VAE_0
def inference():
    model = RNN_VAE_0('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/RNN_VAE_0/L=1e-20_epoch=73-step=107448.ckpt')


if __name__ == '__main__':
    inference()
