
import torch
import numpy as np
import os
import sys
import time

from models import get_il_model
from utils.conf import set_random_seed
from argparse import Namespace
from models.utils.incremental_model import IncrementalModel
from datasets.utils.incremental_dataset import IncrementalDataset
from typing import Tuple
from datasets import get_dataset


def mask_classes(outputs: torch.Tensor, dataset: IncrementalDataset, k: int) -> None:
    cats = dataset.t_c_arr[k]
    outputs[:, 0:cats[0]] = -float('inf')
    outputs[:, cats[-1] + 1:] = -float('inf')


def evaluate(model: IncrementalModel, dataset: IncrementalDataset, last=False) -> Tuple[list, list]:

    accs_taskil, accs_classil, acc_tasks = [], [], []
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue

        correct_k, correct_mask_classes_k, total_k = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            correct_k += torch.sum(pred == labels).item()
            total_k += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
                correct_mask_classes_k += torch.sum(pred == labels).item()

        acc_task = correct_mask_classes_k / total_k * 100
        acc_tasks.append(acc_task)


    accs_taskil.append(correct / total * 100
                if 'class-il' in model.COMPATIBILITY else 0)
    accs_classil.append(correct_mask_classes / total * 100)

    return (accs_taskil, accs_classil), acc_tasks


def train_il(args: Namespace) -> None:
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    dataset = get_dataset(args)
    model = get_il_model(args)

    nt = dataset.nt
    acc_track = []
    for i in range(nt):
        acc_track.append([0.0])

    model.begin_il(dataset)
    mean_accs = []
    for t in range(dataset.nt):

        train_loader, test_loader = dataset.get_data_loaders()

        start_time = time.time()
        model.train_task(dataset, train_loader)
        train_time = time.time() - start_time

        model.test_task(dataset, test_loader)

        start_time = time.time()
        accs, acc_tasks = evaluate(model, dataset)
        test_time = time.time() - start_time


        for i, acc in enumerate(acc_tasks):
            acc_track[i].append(acc)
            print(acc_track[i])

        mean_acc = np.mean(accs, axis=1)
        print_accuracy(mean_acc, t + 1)
        mean_accs.append(mean_acc)

        print('train_time', train_time, 'test_time', test_time)

    model.end_il(dataset)

    for t in range(dataset.nt):
        mean_acc = mean_accs[t]
        print_accuracy(mean_acc, t + 1)

def train_dec(args: Namespace) -> None:
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    dataset = get_dataset(args)
    model = get_il_model(args)

    model.begin_il(dataset)
    mean_accs = []
    for t in range(dataset.nt):

        train_loader, test_loader = dataset.get_data_loaders()

        start_time = time.time()
        model.train_task(dataset, train_loader)
        train_time = time.time() - start_time

        model.test_task(dataset, test_loader)

        start_time = time.time()
        accs, acc_tasks = evaluate(model, dataset)
        test_time = time.time() - start_time


        mean_acc = np.mean(accs, axis=1)
        print_accuracy(mean_acc, t + 1)
        mean_accs.append(mean_acc)

        print('train_time', train_time, 'test_time', test_time)

    # model.end_il(dataset)

    for t in range(dataset.nt):
        mean_acc = mean_accs[t]
        print_accuracy(mean_acc, t + 1)

def print_accuracy(mean_acc: np.ndarray, task_number: int) -> None:

    mean_acc_class_il, mean_acc_task_il = mean_acc
    print('Accuracy for {} task(s): \t [Class-IL]: {} %'
          ' \t [Task-IL]: {} %'.format(task_number, round(
        mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)