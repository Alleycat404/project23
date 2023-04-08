import math
import os

import cv2
import numpy
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Flickr import Flickr
from module import ConvLSTMCell
from module import GDN
from torch.autograd import Variable
import pytorch_lightning as pl
from skimage.metrics import structural_similarity

class RNN_VAE(pl.LightningModule):

    def __init__(self, root='E:/datasets/small_flickr', input_dim=512, h_dim=400, z_dim=20, batchsize=16, img_size=64, L=0.01):
        # 调用父类方法初始化模块的state
        super(RNN_VAE, self).__init__()

        self.L = L
        self.root = root
        self.img_size = img_size
        self.batchsize = batchsize

        self.mse_loss = nn.MSELoss()
        self.KL_loss = nn.KLDivLoss(reduction='batchmean')
        # self.bpp_loss_fn = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.fc1 = nn.Linear(input_dim, h_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(h_dim, z_dim)  # mu
        self.fc3 = nn.Linear(h_dim, z_dim)  # log_var

        # 解码器 ： [b, z_dim] => [b, input_dim]
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, input_dim)

        # 卷积模块
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(512, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # RNN模块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn1 = GDN(64)

        self.rnn_in = nn.LSTM(64, 512, 1, batch_first=True)
        self.register_buffer("h_in", torch.zeros(1, self.batchsize, 512))
        self.register_buffer("c_in", torch.zeros(1, self.batchsize, 512))

        self.conv2 = nn.Conv2d(512, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv3 = nn.Conv2d(32, self.input_dim, kernel_size=1)

        self.rnn_out = nn.LSTM(32, 512, 1, batch_first=True)
        self.register_buffer("h_out", torch.zeros(1, self.batchsize, 512))
        self.register_buffer("c_out", torch.zeros(1, self.batchsize, 512))

        self.conv4 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv5 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.igdn1 = GDN(512, inverse=True)

    def train_dataloader(self):
        train_dataset = Flickr(self.root, "train")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

        return train_loader

    def val_dataloader(self):
        valid_dataset = Flickr(self.root, "test")
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

        return valid_loader

    def test_dataloader(self):
        test_dataset = Flickr(self.root, "test")
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

        return test_loader

    def predict_dataloader(self):
        predict_dataset = Flickr(self.root, "test")
        predict_dataloader = DataLoader(predict_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True)

        return predict_dataloader

    def forward(self, x):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training module [b, batch_size, 1, 28, 28]
        :return: the result of our training module
        """
        batch_size = x.shape[0]  # 每一批含有的样本的个数

        x = self.gdn1(self.conv1(x))  # (1, 3, 64, 64) -> (1, 64, 32, 32)

        # x = x.view(batch_size, -1)  # 一行代表一个样本
        x = x.reshape(batch_size, 64, (self.img_size//2)**2).permute(0, 2, 1)
        x, (self.h_in, self.c_in) = self.rnn_in(x, (self.h_in, self.c_in))  # (b, 1024, 512)
        x = x.permute(0, 2, 1).reshape(batch_size, 512, self.img_size//2, self.img_size//2)  # (b, 512, 32, 32)

        x = self.conv_layer1(x)  # x.shape(b, 32, 4, 4)

        x = x.view(batch_size, -1)
        # encoder
        mu, log_var = self.encoder(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decoder(sampled_z).view(batch_size, 32, self.img_size // 16, self.img_size // 16)

        x_hat = self.conv4(x_hat)  # (b, 512, 4, 4)

        x_hat = F.pixel_shuffle(x_hat, 2)  # (b, 128, 8, 8)
        x_hat = F.pixel_shuffle(x_hat, 2)  # (b, 32, 16, 16)

        x_hat = x_hat.reshape(batch_size, 32, (self.img_size//4)**2).permute(0, 2, 1)
        x_hat, (self.h_out, self.c_out) = self.rnn_out(x_hat, (self.h_out, self.c_out))  # (b, 512, 16, 16)
        x_hat = x_hat.permute(0, 2, 1).reshape(batch_size, 512, self.img_size//4, self.img_size//4)

        x_hat = F.pixel_shuffle(x_hat, 2)  # (b, 128, 32, 32)
        x_hat = F.pixel_shuffle(x_hat, 2)  # (b, 32, 64, 64)

        x_hat = self.conv5(x_hat)

        return x_hat, mu, log_var

    def encoder(self, x):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        h = F.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)

        return mu, log_var

    def reparameterization(self, mu, log_var):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        sigma = torch.exp(log_var * 0.5)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps  # 这里的“*”是点乘的意思

    def decoder(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat

    def training_step(self, batch, idx):
        pred, _, _ = self.forward(batch)
        # print('mse | KL : {} | {}'.format(self.mse_loss(pred, batch), self.KL_loss(F.log_softmax(pred, dim=1), F.softmax(batch, dim=1))))
        loss = self.mse_loss(pred, batch) + self.L * self.KL_loss(F.log_softmax(pred, dim=1), F.softmax(batch, dim=1))
        self.c_in = self.c_in.detach()
        self.h_in = self.h_in.detach()
        self.c_out = self.c_out.detach()
        self.h_out = self.h_out.detach()

        return {'loss': loss, 'image': pred}

    def validation_step(self, batch, idx):
        with torch.no_grad():
            pred, _, _ = self.forward(batch)
            loss = self.mse_loss(pred, batch) + self.L * self.KL_loss(F.log_softmax(pred, dim=1), F.softmax(batch, dim=1))

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # use key 'log'
        self.log('val_loss', avg_loss, prog_bar=True)
        return {'val_loss': avg_loss}

    def test_step(self, batch, idx):
        with torch.no_grad():
            losses = []
            pred, _, _ = self.forward(batch)
            for i in range(pred.shape[0]):
                loss = structural_similarity(
                    pred[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy(),
                    batch[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy(),
                    channel_axis=2)
                losses.append(loss)
            mean_loss = torch.tensor(sum(losses)/len(losses))

        return {'test_loss': mean_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('ssim', avg_loss, prog_bar=True)
        return {'test_loss': avg_loss}

    def predict_step(self, batch, idx, dataloader_idx=0):
        pred, mu, log_var = self.forward(batch)
        sav_pth = 'predict'

        for k in range(pred.shape[0]):
            img = pred[k, :]
            img = img.mul(255).byte()
            img = img.cpu().numpy().transpose(1, 2, 0)
            origin = batch[k, :]
            origin = origin.mul(255).byte()
            origin = origin.cpu().numpy().transpose(1, 2, 0)

            con_img = np.concatenate([origin, img], axis=1)

            # cv2.imshow('{}th image'.format(k), con_img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(sav_pth, str(k)+'.jpg'), con_img)

        loss = self.mse_loss(pred, batch) + self.L * self.KL_loss(F.log_softmax(pred, dim=1), F.softmax(batch, dim=1))
        return pred, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == '__main__':

    root = "E:/datasets/flickr"

    x = torch.rand(16, 3, 64, 64)
    out = RNN_VAE(root).training_step(x, 1)
    print(out['loss'])
