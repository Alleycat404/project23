import pytorch_lightning as pl
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import RNN_VAE, RNN, RNN_CNN, RNN_VAE_1
from Flickr import Flickr

# from prog_bar import GlobalProgressBar

root = "flickr"
img_size = 64

model = RNN_VAE(root, img_size=img_size, L=0.1)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="ckpt", monitor="val_loss")
callbacks = [checkpoint_callback]

trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True, fast_dev_run=False,
                     max_epochs=100, accelerator='auto', check_val_every_n_epoch=2, auto_lr_find=True)

trainer.fit(model)

# print(checkpoint_callback.best_model_path)
