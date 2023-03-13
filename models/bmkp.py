import math
import time
from copy import deepcopy
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.nn.functional import relu, avg_pool2d
from torch.optim.lr_scheduler import StepLR

from models.utils.incremental_model import IncrementalModel

from collections import OrderedDict
import numpy as np


class BMKP(IncrementalModel):
    COMPATIBILITY = ['task-il']
    def __init__(self, args):
        super(BMKP, self).__init__(args)

        self.epochs = args.n_epochs
        self.retrain_epochs = args.retrain_epochs
        self.learning_rate = args.lr
        self.threshold_first = args.threshold_first
        self.threshold = args.threshold
        self.lambd = args.lambd

        self.net = None
        self.loss = F.cross_entropy

        self.feature_list = []
        self.param_weights = []
        self.bn_stats = []

        self.current_task = -1

        self.resnet_size = 0
        self.weight_size = 0
        self.basis_size = 0

    def begin_il(self, dataset):
        self.cpt = int(dataset.nc / dataset.nt)
        self.t_c_arr = dataset.t_c_arr
        self.eye = torch.tril(torch.ones((dataset.nc, dataset.nc))).bool().to(self.device)

        if self.args.dataset == 'seq-tinyimg' or self.args.dataset == 'seq-imagenet':
            self.img_size = 64
        else:
            self.img_size = 32

        if self.args.net == 'LeNet':
            self.net = LeNet(dataset.nt, self.cpt, nf=self.args.nf).to(self.device)
        if self.args.net == 'resnet18':
            self.net = ResNet18(dataset.nc, nf=self.args.nf, nt=dataset.nt, img_size=self.img_size,
                                is_bn_stats=self.args.is_bn_stats).to(self.device)
        if self.args.net == 'resnet34':
            self.net = ResNet34(dataset.nc, nf=self.args.nf, nt=dataset.nt, img_size=self.img_size,
                                is_bn_stats=self.args.is_bn_stats).to(self.device)

        for k, params in enumerate(self.net.get_params()):
            sz = params.data.view(params.data.shape[0], -1).shape
            print('Layer {} - Parameter shape: {}'.format(k + 1, sz))
            self.resnet_size += sz[0] * sz[1]
        print('total size of resnet:', self.resnet_size)

    def train_task(self, dataset, train_loader):
        self.current_task += 1
        self.reset_net(self.net)

        e_sample, e_label = [], []
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            e_sample.append(inputs)
            e_label.append(labels)
            if step > (self.args.example_num / self.args.batch_size): break
        e_sample = torch.cat(e_sample)
        e_label = torch.cat(e_label)

        start_time = time.time()
        self.train_(train_loader, epochs=self.epochs)
        train_time = start_time - time.time()

        start_time = time.time()
        weights = self.decomposition_net(e_sample)
        decompose_time = start_time - time.time()

        start_time = time.time()
        self.composition_net(self.net, weights)
        compose_time = start_time - time.time()

        start_time = time.time()
        self.train_(train_loader, epochs=self.retrain_epochs, is_retrain=True)
        retrain_time = start_time - time.time()

        weights = self.get_weights(self.net)
        self.param_weights.append(weights)
        for i, weight in enumerate(self.param_weights[self.current_task]):
            print('Layer {} - Weight shape: {}'.format(i + 1, weight.shape))
            self.weight_size += weight.shape[0] * weight.shape[1]
        print('total size of weight:', self.weight_size)
        print('train_time:', train_time, 'decompose_time:', decompose_time, 'compose_time:', compose_time, 'retrain_time:', retrain_time)

        self.reset_net(self.net)

    def decomposition_net(self, example_data):
        self.net.eval()

        # forward pass.
        self.net(example_data, self.current_task)

        # extract activations
        if self.args.net == 'LeNet':
            mat_list = self.get_representation_matrix_Lenet(self.net)
        if self.args.net == 'resnet18':
            mat_list = self.get_representation_matrix_ResNet18(self.net)
        if self.args.net == 'resnet34':
            mat_list = self.get_representation_matrix_ResNet34(self.net)

        threshold = self.threshold_first if self.current_task == 0 else self.threshold
        # update basis
        if self.args.is_update_basis or self.current_task == 0:
            print('updating_basis')
            self.feature_list = self.update_Basis(mat_list, threshold, self.feature_list)

        # cal weights of current task
        weights = []
        kk = 0
        for k, params in enumerate(self.net.get_params()):
            sz = params.data.size(0)
            param_weight = torch.mm(
                self.feature_list[kk].transpose(0, 1),
                params.data.view(sz, -1),
            )

            weights.append(param_weight)
            kk += 1

        return weights

    def train_(self, train_loader, epochs, is_retrain=False):
        cls = self.t_c_arr[self.current_task]

        def map_func(x):
            return cls.index(x)

        print('classes', cls)

        self.net.train()

        lf = self.learning_rate if is_retrain else self.learning_rate
        if self.current_task == 0:
            opt = torch.optim.SGD(self.net.parameters(), lr=lf)
        else:
            opt = torch.optim.SGD(self.net.get_params(), lr=lf)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        for epoch in range(int(epochs)):
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1]
                labels = torch.tensor(list(map(map_func, labels))).to(self.device)

                outputs = self.net(inputs, self.current_task)

                loss_ce = self.loss(outputs[:, cls], labels)

                loss_mu = torch.tensor(0.0).to(self.device)
                if not is_retrain and self.current_task > 0:
                    kk = 0
                    for k, params in enumerate(self.net.get_params()):
                        param = params.view(params.data.shape[0], -1)

                        basis = self.feature_list[kk]
                        basis_t = self.feature_list[kk].transpose(0, 1)

                        param_project = torch.mm(basis_t, param)
                        param_sub = param - torch.mm(basis, param_project)

                        param_ = torch.cat([
                            basis_t,
                            param_sub.transpose(0, 1),
                        ], dim=0)

                        corr = torch.mm(
                            param_.transpose(0, 1),
                            param_
                        )
                        loss_mu += (torch.sum(torch.diag(corr)) / torch.sum(corr))

                        kk += 1

                loss = loss_ce + self.lambd * loss_mu

                opt.zero_grad()
                loss.backward()
                opt.step()

            scheduler.step()
            if epoch % self.args.print_freq == 0:
                print('epoch:%d, loss:%.5f, loss_ce:%.5f, loss_mu:%.5f' % (
                    epoch, loss.to('cpu').item(), loss_ce.to('cpu').item(), loss_mu.to('cpu').item()))

    def forward(self, x: torch.Tensor, task_id) -> torch.Tensor:
        weights = self.param_weights[task_id]
        self.composition_net(self.net, weights)
        self.net.eval()

        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x, task_id)
        return outputs


    def origin_forward(self, x: torch.Tensor, task_id):
        self.net.eval()

        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x, task_id)
        return outputs

    def composition_net(self, net, weights):
        conv_list = net.get_convs()

        for idx, conv in enumerate(conv_list):
            weight_conv = nn.Parameter(weights[idx], requires_grad=True)
            valid_dim = weight_conv.shape[0]
            basis_conv = self.feature_list[idx][:, :valid_dim]
            basis_conv = nn.Parameter(basis_conv, requires_grad=False)

            conv.weight_ = weight_conv
            conv.basis_ = basis_conv

    def reset_net(self, net):
        conv_list = net.get_convs()

        for idx, conv in enumerate(conv_list):
            conv.weight_ = None
            conv.basis_ = None

    def get_weights(self, net):
        conv_list = net.get_convs()

        weights = []
        for idx, conv in enumerate(conv_list):
            weights.append(conv.weight_.data)

        return weights

    def update_Basis(self, mat_list, threshold, feature_list=[]):
        if not feature_list:
            for i in range(len(mat_list)):
                activation = mat_list[i].detach().cpu().numpy()
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)

                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
                basis = torch.Tensor(U[:, 0:r]).to(self.device)
                feature_list.append(basis)
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i].detach().cpu().numpy()
                basis = feature_list[i].detach().cpu().numpy()

                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()

                act_hat = activation - np.dot(np.dot(
                    basis,
                    basis.transpose()
                ), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating Basis for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((basis, U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = torch.Tensor(Ui[:, 0:Ui.shape[0]]).to(self.device)
                else:
                    feature_list[i] = torch.Tensor(Ui).to(self.device)

        print('-' * 40)
        print('Basis size Summary')
        print('-' * 40)
        self.basis_size = 0
        for i in range(len(feature_list)):
            print('Layer {} : {}/{}'.format(i + 1, feature_list[i].shape[1], feature_list[i].shape[0]), feature_list[i].shape)
            self.basis_size += feature_list[i].shape[1] * feature_list[i].shape[0]
        print('total basis size:', self.basis_size)
        print('-' * 40)
        return feature_list

    def get_representation_matrix_ResNet18(self, net):
        # Collect activations by forward pass
        act_list = []
        act_list.extend([net.act['conv_in'],
                         net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'],
                         net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
                         net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[0].act['conv_2'],
                         net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
                         net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[0].act['conv_2'],
                         net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
                         net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[0].act['conv_2'],
                         net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])

        mat_final = []
        for act in act_list:
            shape = act.shape
            mat = torch.transpose(act, 0, 1)
            mat = mat.contiguous().view(shape[1], -1)
            mat_final.append(mat)

        return mat_final

    def get_representation_matrix_ResNet34(self, net):
        # Collect activations by forward pass
        act_list = []
        act_list.extend([net.act['conv_in'],
                         net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'],
                         net.layer1[1].act['conv_0'], net.layer1[1].act['conv_1'],
                         net.layer1[2].act['conv_0'], net.layer1[2].act['conv_1'],
                         net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[0].act['conv_2'],
                         net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
                         net.layer2[2].act['conv_0'], net.layer2[2].act['conv_1'],
                         net.layer2[3].act['conv_0'], net.layer2[3].act['conv_1'],
                         net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[0].act['conv_2'],
                         net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
                         net.layer3[2].act['conv_0'], net.layer3[2].act['conv_1'],
                         net.layer3[3].act['conv_0'], net.layer3[3].act['conv_1'],
                         net.layer3[4].act['conv_0'], net.layer3[4].act['conv_1'],
                         net.layer3[5].act['conv_0'], net.layer3[5].act['conv_1'],
                         net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[0].act['conv_2'],
                         net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1'],
                         net.layer4[2].act['conv_0'], net.layer4[2].act['conv_1']])

        mat_final = []
        for act in act_list:
            shape = act.shape
            mat = torch.transpose(act, 0, 1)
            mat = mat.contiguous().view(shape[1], -1)
            mat_final.append(mat)

        return mat_final

    def get_representation_matrix_Lenet(self, net):
        act_list = []
        act_list.extend([
            net.act['conv1'],
            net.act['conv2'],
            net.act['fc1'],
            net.act['fc2'],
        ])

        mat_final = []
        for act in act_list:
            shape = act.shape
            mat = torch.transpose(act, 0, 1)
            mat = mat.contiguous().view(shape[1], -1)
            mat_final.append(mat)

        return mat_final


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def conv3x3(in_planes, out_planes, stride=1):
    return MyConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return MyConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                    padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, nt, stride=1, is_bn_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(planes, track_running_stats=is_bn_stats, affine=True))
        self.bn1s = nn.ModuleList(bn_list)
        self.conv2 = conv3x3(planes, planes)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(planes, track_running_stats=is_bn_stats, affine=True))
        self.bn2s = nn.ModuleList(bn_list)

        self.dropout = nn.Dropout(p=0.1)

        self.shortcut = nn.Sequential()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            bn_list = []
            for i in range(nt):
                bn_list.append(nn.BatchNorm2d(self.expansion * planes, track_running_stats=is_bn_stats, affine=True))
            self.bn3s = nn.ModuleList(bn_list)

            self.conv3 = MyConv2d(in_planes, self.expansion * planes, kernel_size=1,
                                  stride=stride, bias=False)

        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, task_id):
        out = self.conv1(x)
        self.act['conv_0'] = out
        out = relu(self.bn1s[task_id](out))
        out = self.conv2(out)
        self.act['conv_1'] = out
        out = self.bn2s[task_id](out)

        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            sc_out = self.conv3(x)
            self.act['conv_2'] = sc_out
            out += self.bn3s[task_id](sc_out)

        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nc, nt, nf, img_size, is_bn_stats=True):
        super(ResNet, self).__init__()
        self.nt = nt
        self.is_bn_stats = is_bn_stats

        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(nf * 1, track_running_stats=is_bn_stats, affine=True))
        self.bn1s = nn.ModuleList(bn_list)

        self.dropout = nn.Dropout(p=0.1)
        self.blocks = []

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.img_size = img_size
        self.linear = nn.Linear(nf * 8 * block.expansion * int(img_size / 16) * int(img_size / 16), nc, bias=False)
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block_ = block(self.in_planes, planes, self.nt, stride, self.is_bn_stats)
            layers.append(block_)
            self.in_planes = planes * block.expansion
            self.blocks.append(block_)
        return nn.Sequential(*layers)

    def forward(self, x, task_id):
        bsz = x.size(0)
        out = self.conv1(x.view(bsz, 3, self.img_size, self.img_size))
        self.act['conv_in'] = out
        out = relu(self.bn1s[task_id](out))

        for block in self.blocks:
            out = block(out, task_id)

        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y

    def get_params(self) -> torch.Tensor:
        params_arr = []
        for k, (m, params) in enumerate(self.named_parameters()):
            if len(params.size()) == 4:
                params_arr.append(params)
        return params_arr

    def get_convs(self):
        conv_list = [self.conv1]
        for block in self.blocks:
            conv_list.append(block.conv1)
            conv_list.append(block.conv2)
            if hasattr(block, 'conv3'):
                conv_list.append(block.conv3)

        return conv_list


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class LeNet(nn.Module):
    def __init__(self, nt, cpt, nf=10):
        super(LeNet, self).__init__()
        self.nf = nf

        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []

        self.map.append(32)
        self.conv1 = MyConv2d(3, self.nf * 2, kernel_size=5, padding=2, bias=False)

        s = compute_conv_output_size(32, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = MyConv2d(self.nf * 2, self.nf * 5, kernel_size=5, padding=2, bias=False)

        s = compute_conv_output_size(s, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(self.nf * 2)
        self.smid = s
        self.map.append(self.nf * 5 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0)
        self.drop2 = torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1)

        self.fc1 = MyLinear(self.nf * 5 * self.smid * self.smid, self.nf * 80, bias=False)
        self.fc2 = MyLinear(self.nf * 80, self.nf * 50, bias=False)
        self.map.extend([self.nf * 80])

        self.fc3 = torch.nn.Linear(self.nf * 50, cpt * nt, bias=True)

    def forward(self, x, task_id):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        self.act['conv1'] = x
        x = self.maxpool(self.lrn(self.relu(x)))

        x = self.conv2(x)
        self.act['conv2'] = x
        x = self.maxpool(self.lrn(self.relu(x)))

        x = x.reshape(bsz, -1)
        x = self.fc1(x)
        self.act['fc1'] = x
        x = self.relu(x)

        x = self.fc2(x)
        self.act['fc2'] = x
        x = self.relu(x)

        y = self.fc3(x)

        return y

    def get_params(self) -> torch.Tensor:
        params_arr = []
        for k, (m, params) in enumerate(self.named_parameters()):
            if k < 4:
                params_arr.append(params)
        return params_arr

    def get_convs(self):
        return [self.conv1, self.conv2, self.fc1, self.fc2]


def ResNet18(nc, nt, nf=64, img_size=32, is_bn_stats=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nc, nt, nf, img_size, is_bn_stats)


def ResNet34(nc, nt, nf=64, img_size=32, is_bn_stats=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nc, nt, nf, img_size, is_bn_stats)


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


class MyConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kernel_size_ = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(torch.empty(
            (out_channels, in_channels // groups, *kernel_size_), **factory_kwargs))

        self.weight_ = None
        self.basis_ = None

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:

        if self.weight_ is None:
            weight = self.weight
        else:
            shape = self.weight.shape
            weight = torch.mm(self.basis_, self.weight_).view(shape)

        return self._conv_forward(input, weight, self.bias)


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_ = None
        self.basis_ = None
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        if self.weight_ is None:
            weight = self.weight
        else:
            shape = self.weight.shape
            weight = torch.mm(self.basis_, self.weight_).view(shape)

        return F.linear(input, weight, self.bias)
