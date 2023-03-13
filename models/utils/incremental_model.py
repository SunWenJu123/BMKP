
import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


class IncrementalModel(nn.Module):
    """
    Incremental learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, args: Namespace) -> None:
        super(IncrementalModel, self).__init__()

        self.args = args
        self.device = self.args.device

    def begin_il(self, dataset):
        pass

    def train_task(self, dataset, train_loader):
        pass

    def test_task(self, dataset, test_loader):
        pass

    def end_il(self, dataset):
        pass

