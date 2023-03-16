import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import cv2
from torchvision import transforms

class Flickr(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=transforms.ToTensor()):

        assert mode in {"train", "valid", "test"}

        self.mode = mode
        self.root = root
        self.transform = transform
        self.filenames = self._read_filenames()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = cv2.resize(cv2.imread(os.path.join(self.root, self.mode, filename)), (64, 64))

        if self.transform is not None:
            image = self.transform(image)
        return image

    def _read_filenames(self):
        filenames = os.listdir(os.path.join(self.root, self.mode))
        return filenames


if __name__ == '__main__':
    root = "E:/datasets/flickr"
    trainset = Flickr(root, "train")
    print(len(trainset))
