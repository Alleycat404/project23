import pytorch_lightning as pl
from models import RNN_VAE, RNN, RNN_CNN, RNN_VAE_1
def inference():
    model = RNN_VAE('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/RNN_VAE/L=0_epoch=95-step=5952.ckpt')


if __name__ == '__main__':
    inference()
