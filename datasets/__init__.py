
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_mnist import SequentialMNIST
from datasets.cifar100_superclass import Super_CIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils.incremental_dataset import IncrementalDataset
from datasets.seq_imagenet import SequentialImagenet
from argparse import Namespace

NAMES = {
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    Super_CIFAR100.NAME: Super_CIFAR100,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialImagenet.NAME: SequentialImagenet,
}

def get_dataset(args: Namespace) -> IncrementalDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
