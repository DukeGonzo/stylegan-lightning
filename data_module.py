import os
from typing import Optional 
from glob2 import glob

import torch.nn.functional as F
import numpy as np
from PIL import Image as pim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.transforms import RandomHorizontalFlip



torchvision.set_image_backend('accimage')

class DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, 
                       batch_size: int, 
                       resolution: int,
                       balanced_sampling: bool,
                       number_of_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.resolution = resolution
        self.number_of_workers = number_of_workers
        # self.dump_on_disk = dump_on_disk
        # self.dump_path = dump_path
        self.balanced_sampling = balanced_sampling

        self._transforms = transforms.Compose(
            [
                transforms.Resize(resolution), # TODO skip if dump_on_disk 
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ]
        )

        assert os.path.exists(data_dir), f'No such dir {data_dir}'
        self.data_dir = data_dir

    # def prepare_data(self): # TODO: drop it?
    #     if os.path.exists(self.dump_path):
    #         pass

    #     dirs = glob(self.data_dir)
    #     os.mkdir(self.dump_path)

    #     for dir in dirs:
    #         dir_name = dir.split('/')[-1]
    #         os.mkdir(f'{self.dump_path}/{dir_name}')
    #         image_files = glob(dir)

    #         for file in image_files:
    #             file_name = file.split('/')[-1]
    #             pimage = pim.open(file)
    #             pimage = pimage.resize((self.resolution, self.resolution))
    #             pimage = pimage.convert('RGB')
    #             pimage.save(f'{self.dump_path}/{dir_name}/{file_name}')
    

    def setup(self, stage=None):
        self.dataset = im_dataset = ImageFolder(root=self.data_dir, 
        transform=self._transforms)

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

    def train_dataloader(self):
        return DataLoader(self.dataset, 
        batch_size=self.batch_size, 
        drop_last = True,
        sampler=self.sampler, 
        num_workers=self.number_of_workers, 
        pin_memory = True)

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        raise NotImplementedError()
