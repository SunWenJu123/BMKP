import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.conf import base_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.incremental_dataset import IncrementalDataset, store_masked_loaders, getfeature_loader
from datasets.utils.incremental_dataset import get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
from argparse import Namespace
import torchvision.models as models


class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target


class MyTinyImagenet(TinyImagenet):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root: str, train: bool=True, transform: transforms=None,
                target_transform: transforms=None, download: bool=False) -> None:
        self.attributes = []
        self.trans = []
        super(MyTinyImagenet, self).__init__(
            root, train, transform, target_transform, download)

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))
        original_img = img.copy()
        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = (img, target, not_aug_img)
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)

        return ret_tuple


class SequentialTinyImagenet(IncrementalDataset):

    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 200
        self.nt = 10
        self.n_channel = 3
        self.n_imsize1 = 64
        self.n_imsize2 = 64
        super(SequentialTinyImagenet, self).__init__(args)

        if self.args.featureNet:
            self.args.transform = 'pytorch'

            if self.args.featureNet == 'resnet18':
                self.extractor = models.resnet18(pretrained=True)
            elif self.args.featureNet == 'vgg11':
                self.extractor = models.vgg11(pretrained=True)
            elif self.args.featureNet == 'resnet34':
                self.extractor = models.resnet34(pretrained=True)

        if self.args.transform == 'pytorch':
            self.normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.dnormalization_transform = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalization_transform])
            self.test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                self.normalization_transform])
        else:
            self.normalization_transform = transforms.Normalize(mean=[0.4802, 0.4480, 0.3975],
                                                                std=[0.2770, 0.2691, 0.2821])
            self.dnormalization_transform = transforms.Normalize(mean=[0.4802, 0.4480, 0.3975],
                                                                 std=[0.2770, 0.2691, 0.2821])

            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalization_transform])
            self.test_transform = transforms.Compose([transforms.ToTensor(),
                self.normalization_transform])


    def get_data_loaders(self):
        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                 train=True, download=True, transform=self.train_transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    self.test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                        train=False, download=True, transform=self.test_transform)

        train, test = getfeature_loader(train_dataset, test_dataset, setting=self)
        return train, test


    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                            train=True, download=True, transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.train_transform])
        return transform
