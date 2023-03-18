import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model import RNN_VAE
from Flickr import Flickr
from pytorch_lightning.callbacks import RichProgressBar
# from prog_bar import GlobalProgressBar

root = "flickr"

model = RNN_VAE(root)

train_dataset = Flickr(root, "train")
valid_dataset = Flickr(root, "test")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="ckpt", monitor="val_loss")
callbacks = [checkpoint_callback]

trainer = pl.Trainer(callbacks=callbacks, enable_progress_bar=True, fast_dev_run=False,
                     max_epochs=100, accelerator='auto', check_val_every_n_epoch=2, auto_lr_find=True)

trainer.fit(model)

# print(checkpoint_callback.best_model_path)
