import os
from typing import Optional 
from glob2 import glob

import numpy as np
from PIL import Image as pim
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder



torchvision.set_image_backend('accimage')

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, 
                       batch_size: int, 
                       resolution: int,
                       balanced_sampling: bool,
                       dump_on_disk: bool,
                       dump_path: Optional[str] = None):
        super().__init__()
        self.batch_size = batch_size
        self.resolution = resolution
        self.dump_on_disk = dump_on_disk
        self.dump_path = dump_path
        self.balanced_sampling = balanced_sampling

        self._transforms = transforms.Compose(
            [
                transforms.Resize(512), # TODO skip if dump_on_disk 
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        assert os.path.exists(data_dir), f'No such dir {data_dir}'
        self.data_dir = data_dir

    def prepare_data(self): # TODO: drop it?
        if os.path.exists(self.dump_path):
            pass

        dirs = glob(self.data_dir)
        os.mkdir(self.dump_path)

        for dir in dirs:
            dir_name = dir.split('/')[-1]
            os.mkdir(f'{self.dump_path}/{dir_name}')
            image_files = glob(dir)

            for file in image_files:
                fname = file.split('/')[-1]
                pimage = pim.open(file)
                pimage = pimage.resize((self.resolution, self.resolution))
                pimage = pimage.convert('RGB')
                pimage.save(f'{self.dump_path}/{dir_name}/{fname}')
    

    def setup(self, stage=None):
        self.dataset = im_dataset = ImageFolder(root=self.data_dir, transform=self._transforms)

        label_ids = im_dataset.class_to_idx.values()
        targets = np.array(im_dataset.targets)

        label_count = np.zeros(len(label_ids))
        for l in label_ids:
            label_count[l] = np.sum(targets == l)

        label_weight = 1 / (label_count / np.max(label_count))

        weights = np.zeros_like(targets)
        for l, w in enumerate(label_weight):
            weights[targets == l] = w

        sampler = WeightedRandomSampler(weights, len(weights))
        self.sampler = sampler


        # self.mnist_test = MNIST(self.data_dir, train=False)
        # mnist_full = MNIST(self.data_dir, train=True)
        # self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler, num_workers=4)

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()
