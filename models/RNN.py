import math
import os

import cv2
import numpy
import numpy as np
from skimage.metrics import structural_similarity
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Flickr import Flickr
from module import ConvLSTMCell, GDN
from torch.autograd import Variable
import pytorch_lightning as pl


class RNN(pl.LightningModule):

    def __init__(self, root='E:/datasets/small_flickr', batchsize=16, img_size=64):
        # 调用父类方法初始化模块的state
        super(RNN, self).__init__()

        self.loss_fn = nn.MSELoss()
        # self.bpp_loss_fn = torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

        self.batchsize = batchsize

        self.img_size = img_size
        self.hidden1 = (Variable(torch.zeros(batchsize, 256, self.img_size // 4, self.img_size // 4).cuda()),
                        Variable(torch.zeros(batchsize, 256, self.img_size // 4, self.img_size // 4).cuda()))
        self.hidden2 = (Variable(torch.zeros(batchsize, 512, self.img_size // 8, self.img_size // 8).cuda()),
                        Variable(torch.zeros(batchsize, 512, self.img_size // 8, self.img_size // 8).cuda()))
        self.hidden3 = (Variable(torch.zeros(batchsize, 512, self.img_size // 16, self.img_size // 16).cuda()),
                        Variable(torch.zeros(batchsize, 512, self.img_size // 16, self.img_size // 16).cuda()))
        self.hidden4 = (Variable(torch.zeros(batchsize, 512, self.img_size // 16, self.img_size // 16).cuda()),
                        Variable(torch.zeros(batchsize, 512, self.img_size // 16, self.img_size // 16).cuda()))
        self.hidden5 = (Variable(torch.zeros(batchsize, 512, self.img_size // 8, self.img_size // 8).cuda()),
                        Variable(torch.zeros(batchsize, 512, self.img_size // 8, self.img_size // 8).cuda()))
        self.hidden6 = (Variable(torch.zeros(batchsize, 256, self.img_size // 4, self.img_size // 4).cuda()),
                        Variable(torch.zeros(batchsize, 256, self.img_size // 4, self.img_size // 4).cuda()))
        self.hidden7 = (Variable(torch.zeros(batchsize, 128, self.img_size // 2, self.img_size // 2).cuda()),
                        Variable(torch.zeros(batchsize, 128, self.img_size // 2, self.img_size // 2).cuda()))

        # encoder

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.gdn1 = GDN(64)

        # self.conv2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.gdn2 = GDN(64)

        # self.relu = nn.LeakyReLU()

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

        # binarizer

        self.conv2 = nn.Conv2d(512, 32, kernel_size=1, bias=False)

        # decoder

        self.conv3 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.igdn = GDN(512, inverse=True)

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

        self.root = root

    def train_dataloader(self):
        train_dataset = Flickr(self.root, "train")
        train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=0, drop_last=True)

        return train_loader

    def val_dataloader(self):
        valid_dataset = Flickr(self.root, "test")
        valid_loader = DataLoader(valid_dataset, batch_size=self.batchsize, shuffle=False, num_workers=0, drop_last=True)

        return valid_loader

    def test_dataloader(self):
        test_dataset = Flickr(self.root, "test")
        test_loader = DataLoader(test_dataset, batch_size=self.batchsize, shuffle=False, num_workers=0, drop_last=True)

        return test_loader

    def predict_dataloader(self):
        predict_dataset = Flickr(self.root, "test")
        predict_dataloader = DataLoader(predict_dataset, batch_size=self.batchsize, shuffle=False, num_workers=0, drop_last=True)

        return predict_dataloader

    def forward(self, input):
        # batch_size = input.shape[0]  # 每一批含有的样本的个数

        # encoer
        x = self.gdn1(self.conv1(input))

        self.hidden1 = self.rnn1(x, self.hidden1)
        x = self.hidden1[0]

        self.hidden2 = self.rnn2(x, self.hidden2)
        x = self.hidden2[0]

        self.hidden3 = self.rnn3(x, self.hidden3)
        x = self.hidden3[0]

        # binarizer
        feat = self.conv2(x)
        x = torch.tanh(feat)

        # decoder
        x = self.igdn(self.conv3(x))

        self.hidden4 = self.rnn4(x, self.hidden4)
        x = self.hidden4[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden5 = self.rnn5(x, self.hidden5)
        x = self.hidden5[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden6 = self.rnn6(x, self.hidden6)
        x = self.hidden6[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden7 = self.rnn7(x, self.hidden7)
        x = self.hidden7[0]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.conv4(x)) / 2

        return x

    def encode(self, input):
        """
        encoding part
        :param x: input image
        :return: mu and log_var
        """
        x = self.gdn1(self.conv1(input))

        self.hidden1 = self.rnn1(x, self.hidden1)
        x = self.hidden1[0]

        self.hidden2 = self.rnn2(x, self.hidden2)
        x = self.hidden2[0]

        self.hidden3 = self.rnn3(x, self.hidden3)
        x = self.hidden3[0]

        return x

    def binarizer(self, input):
        feat = self.conv2(input)
        x = torch.tanh(feat)
        return x

    def decode(self, input):
        x = self.igdn(self.conv3(input))

        self.hidden4 = self.rnn4(x, self.hidden4)
        x = self.hidden4[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden5 = self.rnn5(x, self.hidden5)
        x = self.hidden5[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden6 = self.rnn6(x, self.hidden6)
        x = self.hidden6[0]
        x = F.pixel_shuffle(x, 2)

        self.hidden7 = self.rnn7(x, self.hidden7)
        x = self.hidden7[0]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.conv4(x)) / 2

        return x

    def training_step(self, batch, idx):
        pred = self.forward(batch)
        loss = self.loss_fn(pred, batch)

        self.hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        self.hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())
        self.hidden3 = (self.hidden3[0].detach(), self.hidden3[1].detach())
        self.hidden4 = (self.hidden4[0].detach(), self.hidden4[1].detach())
        self.hidden5 = (self.hidden5[0].detach(), self.hidden5[1].detach())
        self.hidden6 = (self.hidden6[0].detach(), self.hidden6[1].detach())
        self.hidden7 = (self.hidden7[0].detach(), self.hidden7[1].detach())

        return {'loss': loss, 'image': pred}

    def validation_step(self, batch, idx):
        with torch.no_grad():
            pred = self.forward(batch)
            loss = self.loss_fn(pred, batch)

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
            pred = self.forward(batch)
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
        self.log('test_loss', avg_loss, prog_bar=True)
        return {'test_loss': avg_loss}

    def predict_step(self, batch, idx, dataloader_idx=0):
        pred = self.forward(batch)
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
            cv2.imwrite(os.path.join(sav_pth, str(k) + '.jpg'), con_img)

        loss = self.loss_fn(pred, batch)
        return pred, loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


if __name__ == '__main__':
    root = "E:/datasets/flickr"

    img_size = 64
    x = torch.rand(16, 3, img_size, img_size).cuda()

    model = RNN(root).cuda()
    out = model(x)
    print(out.shape)
