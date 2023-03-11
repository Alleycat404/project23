import math

import cv2
import numpy
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Flickr import Flickr
from module import ConvLSTMCell
from module import GDN
from torch.autograd import Variable
import pytorch_lightning as pl

class RNN_VAE(pl.LightningModule):

    def __init__(self, root='E:/datasets/small_flickr', input_dim=512, h_dim=400, z_dim=20):
        # 调用父类方法初始化模块的state
        super(RNN_VAE, self).__init__()

        self.loss_fn = nn.MSELoss()
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

        # RNN模块
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn1 = GDN(64)

        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.conv2 = nn.Conv2d(512, 32, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Conv2d(32, self.input_dim, kernel_size=1)

        self.rnn4 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn5 = ConvLSTMCell(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

        self.rnn6 = ConvLSTMCell(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.rnn7 = ConvLSTMCell(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)

        self.conv4 = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.igdn1 = GDN(512, inverse=True)
        self.root = root

    def train_dataloader(self):
        train_dataset = Flickr(self.root, "train")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

        return train_loader

    def val_dataloader(self):
        valid_dataset = Flickr(self.root, "test")
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=0)

        return valid_loader

    def test_dataloader(self):
        test_dataset = Flickr(self.root, "test")
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

        return test_loader

    def predict_dataloader(self):
        predict_dataloader = Flickr(self.root, "test")
        predict_dataloader = DataLoader(predict_dataloader, batch_size=2, shuffle=False, num_workers=0)

        return predict_dataloader

    def forward(self, x, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7):
        """
        向前传播部分, 在model_name(inputs)时自动调用
        :param x: the input of our training module [b, batch_size, 1, 28, 28]
        :return: the result of our training module
        """

        x, hidden1, hidden2, hidden3 = self.RNN_encode(x, hidden1, hidden2, hidden3)  # x.shape(1, 32, 4, 4)
        batch_size = x.shape[0]  # 每一批含有的样本的个数
        x = x.view(batch_size, self.input_dim)  # 一行代表一个样本

        # encoder
        mu, log_var = self.encode(x)
        # reparameterization trick
        sampled_z = self.reparameterization(mu, log_var)
        # decoder
        x_hat = self.decode(sampled_z).view(batch_size, 32, 4, 4)

        x_hat, hidden4, hidden5, hidden6, hidden7 = self.RNN_decode(x_hat, hidden4, hidden5, hidden6, hidden7)
        # reshape
        x_hat = x_hat.view(batch_size, 3, 64, 64)
        return x_hat, mu, log_var

    def encode(self, x):
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

    def decode(self, z):
        """
        Given a sampled z, decode it back to image
        :param z:
        :return:
        """
        h = F.relu(self.fc4(z))
        x_hat = torch.sigmoid(self.fc5(h))  # 图片数值取值为[0,1]，不宜用ReLU
        return x_hat

    def RNN_encode(self, input, hidden1, hidden2, hidden3):
        x = self.gdn1(self.conv1(input))  # (1, 3, 64, 64) -> (1, 64, 32, 32)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]  # (1, 64, 32, 32) -> (1, 256, 16, 16)

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]  # (1, 256, 16, 16) -> (1, 512, 8, 8)

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]  # x channel = 512 y.shape = (1, 512, 4, 4)

        x = self.conv2(x)  # x.shape(1, 32, 4, 4)

        return x, hidden1, hidden2, hidden3

    def RNN_decode(self, input, hidden1, hidden2, hidden3, hidden4):
        x = self.igdn1(self.conv3(input))  # (1, 512, 4, 4)

        hidden1 = self.rnn4(x, hidden1)
        x = hidden1[0]  # (1, 512, 4, 4)
        x = F.pixel_shuffle(x, 2)  # (1, 128, 8, 8)

        hidden2 = self.rnn5(x, hidden2)
        x = hidden2[0]  # (1, 512, 8, 8)
        x = F.pixel_shuffle(x, 2)  # (1, 128, 16, 16)

        hidden3 = self.rnn6(x, hidden3)
        x = hidden3[0]  # (1, 256, 16, 16)
        x = F.pixel_shuffle(x, 2)  # (1, 64, 32, 32)

        hidden4 = self.rnn7(x, hidden4)
        x = hidden4[0]  # (1, 128, 32, 32)
        x = F.pixel_shuffle(x, 2)  # (1, 32, 64, 64)

        x = torch.tanh(self.conv4(x)) / 2

        return x, hidden1, hidden2, hidden3, hidden4

    def training_step(self, batch, idx):
        x_h_1 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_2 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_3 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))

        x_h_4 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))
        x_h_5 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_6 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_7 = (Variable(torch.zeros(1, 128, 32, 32).cuda()),
                 Variable(torch.zeros(1, 128, 32, 32).cuda()))

        # x_h_1 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_2 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_3 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        #
        # x_h_4 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        # x_h_5 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_6 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_7 = (Variable(torch.zeros(1, 128, 32, 32)),
        #          Variable(torch.zeros(1, 128, 32, 32)))
        pred, _, _ = self.forward(batch, x_h_1, x_h_2, x_h_3, x_h_4, x_h_5, x_h_6, x_h_7)
        loss = self.loss_fn(pred, batch)

        return {'loss': loss, 'image': pred}

    def validation_step(self, batch, idx):
        x_h_1 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_2 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_3 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))

        x_h_4 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))
        x_h_5 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_6 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_7 = (Variable(torch.zeros(1, 128, 32, 32).cuda()),
                 Variable(torch.zeros(1, 128, 32, 32).cuda()))

        # x_h_1 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_2 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_3 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        #
        # x_h_4 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        # x_h_5 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_6 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_7 = (Variable(torch.zeros(1, 128, 32, 32)),
        #          Variable(torch.zeros(1, 128, 32, 32)))

        with torch.no_grad():
            pred, _, _ = self.forward(batch, x_h_1, x_h_2, x_h_3, x_h_4, x_h_5, x_h_6, x_h_7)
            loss = self.loss_fn(pred, batch)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # use key 'log'
        self.log('val_loss', avg_loss, prog_bar=True)
        return {'val_loss': avg_loss}

    def test_step(self, batch, idx):
        x_h_1 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_2 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_3 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))

        x_h_4 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))
        x_h_5 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_6 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_7 = (Variable(torch.zeros(1, 128, 32, 32).cuda()),
                 Variable(torch.zeros(1, 128, 32, 32).cuda()))

        # x_h_1 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_2 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_3 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        #
        # x_h_4 = (Variable(torch.zeros(1, 512, 4, 4)),
        #          Variable(torch.zeros(1, 512, 4, 4)))
        # x_h_5 = (Variable(torch.zeros(1, 512, 8, 8)),
        #          Variable(torch.zeros(1, 512, 8, 8)))
        # x_h_6 = (Variable(torch.zeros(1, 256, 16, 16)),
        #          Variable(torch.zeros(1, 256, 16, 16)))
        # x_h_7 = (Variable(torch.zeros(1, 128, 32, 32)),
        #          Variable(torch.zeros(1, 128, 32, 32)))

        with torch.no_grad():
            pred, _, _ = self.forward(batch, x_h_1, x_h_2, x_h_3, x_h_4, x_h_5, x_h_6, x_h_7)
            loss = self.loss_fn(pred, batch)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss, prog_bar=True)
        return {'test_loss': avg_loss}

    def predict_step(self, batch, idx, dataloader_idx=0):
        x_h_1 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_2 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_3 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))

        x_h_4 = (Variable(torch.zeros(1, 512, 4, 4).cuda()),
                 Variable(torch.zeros(1, 512, 4, 4).cuda()))
        x_h_5 = (Variable(torch.zeros(1, 512, 8, 8).cuda()),
                 Variable(torch.zeros(1, 512, 8, 8).cuda()))
        x_h_6 = (Variable(torch.zeros(1, 256, 16, 16).cuda()),
                 Variable(torch.zeros(1, 256, 16, 16).cuda()))
        x_h_7 = (Variable(torch.zeros(1, 128, 32, 32).cuda()),
                 Variable(torch.zeros(1, 128, 32, 32).cuda()))
        # take average of `self.mc_iteration` iterations
        pred, _, _ = self.forward(batch, x_h_1, x_h_2, x_h_3, x_h_4, x_h_5, x_h_6, x_h_7)

        for k in range(pred.shape[0]):
            img = pred[k, :]
            img = img.mul(255).byte()
            img = img.cpu().numpy().transpose((1, 2, 0))

            cv2.imshow('{}th image'.format(k), img)
            key = cv2.waitKey(0)

        loss = self.loss_fn(pred, batch)
        return pred, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == '__main__':

    root = "E:/datasets/flickr"

    x_h_1 = (Variable(torch.zeros(1, 256, 16, 16)),
             Variable(torch.zeros(1, 256, 16, 16)))
    x_h_2 = (Variable(torch.zeros(1, 512, 8, 8)),
             Variable(torch.zeros(1, 512, 8, 8)))
    x_h_3 = (Variable(torch.zeros(1, 512, 4, 4)),
             Variable(torch.zeros(1, 512, 4, 4)))

    x_h_4 = (Variable(torch.zeros(1, 512, 4, 4)),
             Variable(torch.zeros(1, 512, 4, 4)))
    x_h_5 = (Variable(torch.zeros(1, 512, 8, 8)),
             Variable(torch.zeros(1, 512, 8, 8)))
    x_h_6 = (Variable(torch.zeros(1, 256, 16, 16)),
             Variable(torch.zeros(1, 256, 16, 16)))
    x_h_7 = (Variable(torch.zeros(1, 128, 32, 32)),
             Variable(torch.zeros(1, 128, 32, 32)))
    x = torch.rand(1, 3, 64, 64)
    model = RNN_VAE(root)
    out, mu, log_var = model(x, x_h_1, x_h_2, x_h_3, x_h_4, x_h_5, x_h_6, x_h_7)
    print(out.shape)
