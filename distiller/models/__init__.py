#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This package contains ImageNet and CIFAR image classification models for pytorch"""

import copy
from functools import partial
import torch
import torchvision.models as torch_models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn as nn
from . import cifar10 as cifar10_models
from . import mnist as mnist_models
from . import imagenet as imagenet_extra_models
import pretrainedmodels

from distiller.utils import set_model_input_shape_attr, model_setattr
from distiller.modules import Mean, EltwiseAdd

import logging

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import get_cfg

import numpy as np
import pprint
import torch
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.models import build_model
import slowfast.utils.checkpoint as cu

cfg = get_cfg()
cfg.merge_from_file("/workspace/Kugos/distiller-3d/SlowFast/configs/SLOWFAST_8x8_R50-UCF101.yaml")
# cfg.NUM_GPUS = 1
# cfg.TRAIN.BATCH_SIZE = 2
cfg.DATA.PATH_TO_DATA_DIR = "/workspace/Data/"

msglogger = logging.getLogger()

SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'mnist', 'ucf101', 'slowfast_ucf101')

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

IMAGENET_MODEL_NAMES = copy.deepcopy(TORCHVISION_MODEL_NAMES)
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))
IMAGENET_MODEL_NAMES.extend(pretrainedmodels.model_names)

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

MNIST_MODEL_NAMES = sorted(name for name in mnist_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(mnist_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES + MNIST_MODEL_NAMES)))

ALL_MODEL_NAMES.append("c3d_ucf101")
ALL_MODEL_NAMES.append("slowfast_ucf101")

# print("ALL: ", ALL_MODEL_NAMES)

# print()

class C3D(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=True):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)
        
        print("*"*10, " output for fc ", "*"*10)
        print(logits[0])

        probs = nn.Softmax(dim=1)(logits)
        return probs
    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "conv1.weight": "conv1.weight",
                        "conv1.bias": "conv1.bias",
                        # Conv2
                        "conv2.weight": "conv2.weight",
                        "conv2.bias": "conv2.bias",
                        # Conv3a
                        "conv3a.weight": "conv3a.weight",
                        "conv3a.bias": "conv3a.bias",
                        # Conv3b
                        "conv3b.weight": "conv3b.weight",
                        "conv3b.bias": "conv3b.bias",
                        # Conv4a
                        "conv4a.weight": "conv4a.weight",
                        "conv4a.bias": "conv4a.bias",
                        # Conv4b
                        "conv4b.weight": "conv4b.weight",
                        "conv4b.bias": "conv4b.bias",
                        # Conv5a
                        "conv5a.weight": "conv5a.weight",
                        "conv5a.bias": "conv5a.bias",
                         # Conv5b
                         "conv5b.weight": "conv5b.weight",
                        "conv5b.bias": "conv5b.bias",
                        # fc6
                        "fc6.weight": "fc6.weight",
                        "fc6.bias": "fc6.bias",
                        # fc7
                        "fc7.weight": "fc7.weight",
                        "fc7.bias": "fc7.bias",
                        }

        p_dict = torch.load('/workspace/Kugos/distiller-3d/c3d_sport_1m.pickle')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def patch_torchvision_mobilenet_v2(model):
    """
    Patches TorchVision's MobileNetV2:
    * To allow quantization, this adds modules for tensor operations (mean, element-wise addition) to the
      model instance and patches the forward functions accordingly
    * Fixes a bug in the torchvision implementation that prevents export to ONNX (and creation of SummaryGraph)
    """
    if not isinstance(model, torch_models.MobileNetV2):
        raise TypeError("Only MobileNetV2 is acceptable.")

    def patched_forward_mobilenet_v2(self, x):
        x = self.features(x)
        # x = x.mean([2, 3]) # this was a bug: https://github.com/pytorch/pytorch/issues/20516
        x = self.mean32(x)
        x = self.classifier(x)
        return x
    model.mean32 = nn.Sequential(
        Mean(3), Mean(2)
    )
    model.__class__.forward = patched_forward_mobilenet_v2

    def is_inverted_residual(module):
        return isinstance(module, nn.Module) and module.__class__.__name__ == 'InvertedResidual'

    def patched_forward_invertedresidual(self, x):
        if self.use_res_connect:
            return self.residual_eltwiseadd(self.conv(x), x)
        else:
            return self.conv(x)

    for n, m in model.named_modules():
        if is_inverted_residual(m):
            if m.use_res_connect:
                m.residual_eltwiseadd = EltwiseAdd()
            m.__class__.forward = patched_forward_invertedresidual


_model_extensions = {}


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported)
        arch: architecture name
        parallel [boolean]: if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'imagenet':
            model, cadene = _create_imagenet_model(arch, pretrained)
        elif dataset == 'cifar10':
            model = _create_cifar10_model(arch, pretrained)
        elif dataset == 'mnist':
            model = _create_mnist_model(arch, pretrained)
        elif dataset == 'ucf101':
            model = _create_ucf101_model(arch, pretrained)
        elif dataset == 'slowfast_ucf101':
            model = _create_slowfast_ucf101_model(arch, pretrained)
    except ValueError:
        if _is_registered_extension(arch, dataset, pretrained):
            model = _create_extension_model(arch, dataset)
        else:
            raise ValueError('Could not recognize dataset {} and arch {} pair'.format(dataset, arch))

    msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                     arch, dataset))
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.is_parallel = parallel
    else:
        device = 'cpu'
        model.is_parallel = False

    # Cache some attributes which describe the model
    _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)
    model.arch = arch
    model.dataset = dataset
    return model.to(device)

def _create_ucf101_model(arch, pretrained):
    dataset = "ucf101"

    if "c3d" in arch:
        model = C3D(101)

    return model

def _create_slowfast_ucf101_model(arch, pretrained):
    dataset = "slowfast_ucf101"
    
    # print("slowfast ucf101 model called: ")

    if "slowfast" in arch:
        model = build_model(cfg)
        cu.load_checkpoint(cfg.TRAIN.CHECKPOINT_FILE_PATH, model, False, None, False, convert_from_caffe2=True)
        
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True
            
        # print("slowfast model: ", model)
    
    return model
    
def _create_imagenet_model(arch, pretrained):
    dataset = "imagenet"
    cadene = False
    model = None
    if arch in RESNET_SYMS:
        model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
    elif arch in TORCHVISION_MODEL_NAMES:
        try:
            model = getattr(torch_models, arch)(pretrained=pretrained)
            if arch == "mobilenet_v2":
                patch_torchvision_mobilenet_v2(model)
        except NotImplementedError:
            # In torchvision 0.3, trying to download a model that has no
            # pretrained image available will raise NotImplementedError
            if not pretrained:
                raise
    if model is None and (arch in imagenet_extra_models.__dict__) and not pretrained:
        model = imagenet_extra_models.__dict__[arch]()
    if model is None and (arch in pretrainedmodels.model_names):
        cadene = True
        model = pretrainedmodels.__dict__[arch](
            num_classes=1000,
            pretrained=(dataset if pretrained else None))
    if model is None:
        error_message = ''
        if arch not in IMAGENET_MODEL_NAMES:
            error_message = "Model {} is not supported for dataset ImageNet".format(arch)
        elif pretrained:
            error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
        raise ValueError(error_message or 'Failed to find model {}'.format(arch))
    return model, cadene


def _create_cifar10_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_mnist_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (MNIST) does not have a pretrained model".format(arch))
    try:
        model = mnist_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset MNIST".format(arch))
    return model


def _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene):
    if cadene and pretrained:
        # When using pre-trained weights, Cadene models already have an input size attribute
        # We add the batch dimension to it
        input_size = model.module.input_size if isinstance(model, torch.nn.DataParallel) else model.input_size
        shape = tuple([1] + input_size)
        set_model_input_shape_attr(model, input_shape=shape)
    elif arch == 'inception_v3':
        set_model_input_shape_attr(model, input_shape=(1, 3, 299, 299))
    else:
        set_model_input_shape_attr(model, dataset=dataset)


def register_user_model(arch, dataset, model):
    """A simple mechanism to support models that are not part of distiller.models"""
    _model_extensions[(arch, dataset)] = model


def _is_registered_extension(arch, dataset, pretrained):
    try:
        return _model_extensions[(arch, dataset)] is not None
    except KeyError:
        return False


def _create_extension_model(arch, dataset):
    return _model_extensions[(arch, dataset)]()