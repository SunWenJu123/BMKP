from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP import MNISTMLP
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
import numpy as np
from datasets.utils.validation import get_train_val
from datasets.utils.incremental_dataset import store_masked_loaders, IncrementalDataset, getfeature_loader
from typing import Tuple
from argparse import Namespace


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        self.attributes = []
        self.trans = []
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)
        self.transform = transform

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index: int):
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = (img, target, original_img)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)

        return ret_tuple



class SequentialMNIST(IncrementalDataset):

    NAME = 'seq-mnist'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = None
    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 10
        self.nt = 5
        super(SequentialMNIST, self).__init__(args)


    def get_data_loaders(self):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                train=False, download=True, transform=transform)

        train, test = getfeature_loader(train_dataset, test_dataset, self)
        return train, test


    def not_aug_dataloader(self, batch_size):
        transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i -
            self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader

