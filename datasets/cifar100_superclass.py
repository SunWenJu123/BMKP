
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.utils import shuffle

import pickle
from argparse import Namespace

from datasets.utils.incremental_dataset import ILDataset, IncrementalDataset


def cifar100_superclass_python(task_order, group=5, validation=False, val_ratio=0.05, flat=False, one_hot=True, seed = 0 ):
    CIFAR100_LABELS_LIST = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]

    sclass = []
    sclass.append(' beaver, dolphin, otter, seal, whale,')  # aquatic mammals
    sclass.append(' aquarium_fish, flatfish, ray, shark, trout,')  # fish
    sclass.append(' orchid, poppy, rose, sunflower, tulip,')  # flowers
    sclass.append(' bottle, bowl, can, cup, plate,')  # food
    sclass.append(' apple, mushroom, orange, pear, sweet_pepper,')  # fruit and vegetables
    sclass.append(' clock, computer keyboard, lamp, telephone, television,')  # household electrical devices
    sclass.append(' bed, chair, couch, table, wardrobe,')  # household furniture
    sclass.append(' bee, beetle, butterfly, caterpillar, cockroach,')  # insects
    sclass.append(' bear, leopard, lion, tiger, wolf,')  # large carnivores
    sclass.append(' bridge, castle, house, road, skyscraper,')  # large man-made outdoor things
    sclass.append(' cloud, forest, mountain, plain, sea,')  # large natural outdoor scenes
    sclass.append(' camel, cattle, chimpanzee, elephant, kangaroo,')  # large omnivores and herbivores
    sclass.append(' fox, porcupine, possum, raccoon, skunk,')  # medium-sized mammals
    sclass.append(' crab, lobster, snail, spider, worm,')  # non-insect invertebrates
    sclass.append(' baby, boy, girl, man, woman,')  # people
    sclass.append(' crocodile, dinosaur, lizard, snake, turtle,')  # reptiles
    sclass.append(' hamster, mouse, rabbit, shrew, squirrel,')  # small mammals
    sclass.append(' maple_tree, oak_tree, palm_tree, pine_tree, willow_tree,')  # trees
    sclass.append(' bicycle, bus, motorcycle, pickup_truck, train,')  # vehicles 1
    sclass.append(' lawn_mower, rocket, streetcar, tank, tractor,')  # vehicles 2

    # download CIFAR100
    dataset_train = datasets.CIFAR100('./data/CIFAR100/' ,train=True, download=True)
    dataset_test  = datasets.CIFAR100('./data/CIFAR100/' ,train=False ,download=True)

    if validation == True:
        data_path = './data/CIFAR100/cifar-100-python/train'
    else:
        data_path = './data/CIFAR100/cifar-100-python/test'

    n_classes = 100
    size =[3 ,32 ,32]
    data ={}
    taskcla =[]
    mean = np.array([x / 255 for x in [125.3, 123.0, 113.9]])
    std = np.array([x / 255 for x in [63.0, 62.1, 66.7]])

    files = open(data_path, 'rb')
    dict = pickle.load(files, encoding='bytes')

    # NOTE Image Standardization
    images = (dict[b'data'])
    images = np.float32(images) / 255
    labels = dict[b'fine_labels']
    labels_pair = [[jj for jj in range(100) if ' %s,' % CIFAR100_LABELS_LIST[jj] in sclass[kk]] for kk in range(20)]

    # flat_pair = np.concatenate(labels_pair)

    argsort_sup = [[] for _ in range(20)]
    for _i in range(len(images)):
        for _j in range(20):
            if labels[_i] in labels_pair[_j]:
                argsort_sup[_j].append(_i)

    argsort_sup_c = np.concatenate(argsort_sup)

    train_split = []
    val_split = []
    position = [_k for _k in range(0, len(images) + 1, int(len(images) / 20))]

    if validation == True:
        s_train = 'train'
        s_valid = 'valid'
    else:
        s_train = 'test'

    for idx in task_order:
        data[idx] = {}
        data[idx]['name'] = 'cifar100'
        data[idx]['ncla'] = 5
        data[idx][s_train] = {'x': [], 'y': []}
        # print('range : [%d,%d]'%(position[idx], position[idx+1]))
        gimages = np.take(images, argsort_sup_c[position[idx]:position[idx + 1]], axis=0)

        if not flat:
            gimages = gimages.reshape([gimages.shape[0], 32, 32, 3])

            # gimages = (gimages-mean)/std # mean,std normalization
            gimages = gimages.swapaxes(2, 3).swapaxes(1, 2)
            # gimages = tf.image.per_image_standardization(gimages)

        glabels = np.take(labels, argsort_sup_c[position[idx]:position[idx + 1]])
        for _si, swap in enumerate(labels_pair[idx]):
            glabels = ['%d' % _si if x == swap else x for x in glabels]
        # if idx <2:
        #     imshow(gimages[0])

        data[idx][s_train]['x'] = torch.FloatTensor(gimages)

        data[idx][s_train]['y'] = torch.LongTensor(np.array([np.int32(glabels)], dtype=int)).view(-1)
        # print(data[idx][s_train]['x'].max(), data[idx][s_train]['x'].min())

        if validation == True:
            r = np.arange(data[idx][s_train]['x'].size(0))
            r = np.array(shuffle(r, random_state=seed), dtype=int)
            nvalid = int(val_ratio * len(r))
            ivalid = torch.LongTensor(r[:nvalid])
            itrain = torch.LongTensor(r[nvalid:])
            data[idx]['valid'] = {}
            data[idx]['valid']['x'] = data[idx]['train']['x'][ivalid].clone()
            data[idx]['valid']['y'] = data[idx]['train']['y'][ivalid].clone()
            data[idx]['train']['x'] = data[idx]['train']['x'][itrain].clone()
            data[idx]['train']['y'] = data[idx]['train']['y'][itrain].clone()
    # pdb.set_trace()
    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla


class Super_CIFAR100(IncrementalDataset):
    NAME = 'sup-cifar100'
    SETTING = 'class-il'
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.nc = 100
        self.nt = 20
        self.n_channel = 3
        self.n_imsize1 = 32
        self.n_imsize2 = 32
        super(Super_CIFAR100, self).__init__(args)

        task_order = [np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                      np.array([15, 12, 5, 9, 7, 16, 18, 17, 1, 0, 3, 8, 11, 14, 10, 6, 2, 4, 13, 19]),
                      np.array([17, 1, 19, 18, 12, 7, 6, 0, 11, 15, 10, 5, 13, 3, 9, 16, 4, 14, 2, 8]),
                      np.array([11, 9, 6, 5, 12, 4, 0, 10, 13, 7, 14, 3, 15, 16, 8, 1, 2, 19, 18, 17]),
                      np.array([6, 14, 0, 11, 12, 17, 13, 4, 9, 1, 7, 19, 8, 10, 3, 15, 18, 5, 2, 16])]
        self.data, taskcla = cifar100_superclass_python(task_order[0], group=5, validation=True)
        self.test_data, _ = cifar100_superclass_python(task_order[0], group=5)

    def get_data_loaders(self):

        xtrain=self.data[self.i]['train']['x']
        ytrain=self.data[self.i]['train']['y'] + self.i * 5
        xvalid=self.data[self.i]['valid']['x']
        yvalid=self.data[self.i]['valid']['y'] + self.i * 5
        xtest =self.test_data[self.i]['test']['x']
        ytest =self.test_data[self.i]['test']['y'] + self.i * 5

        train_dataset = ILDataset(xtrain, ytrain)
        test_dataset = ILDataset(xtest, ytest)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        self.i += 1

        return train_loader, test_loader

