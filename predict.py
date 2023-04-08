import pytorch_lightning as pl
from models import RNN_VAE, RNN, RNN_CNN
def inference():
    model = RNN('small_flickr')
    trainer = pl.Trainer(accelerator='auto')
    # trainer.fit(model)
    trainer.predict(model, ckpt_path='ckpt/RNN/epoch=39-step=58080.ckpt')


if __name__ == '__main__':
    inference()
