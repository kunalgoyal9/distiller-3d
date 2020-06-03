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

"""Helper code for data loading.

This code will help with the image classification datasets: ImageNet and CIFAR10

"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import Sampler
import numpy as np
import distiller
from sklearn.model_selection import train_test_split

import cv2
from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from slowfast.datasets import loader
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import get_cfg

cfg = get_cfg()
cfg.merge_from_file("/workspace/Kugos/distiller-3d/SlowFast/configs/SLOWFAST_8x8_R50-UCF101.yaml")
# cfg.NUM_GPUS = 1
# cfg.TRAIN.BATCH_SIZE = 2
cfg.DATA.PATH_TO_DATA_DIR = "/workspace/Data/"


DATASETS_NAMES = ['imagenet', 'cifar10', 'mnist', 'ucf101', 'slowfast_ucf101']


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, data_dir=""):
        self.root_dir = data_dir
        self.output_dir = data_dir + "_segregated"
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # print("*****", self.check_preprocess())

        if (not self.check_preprocess()) or preprocess:
            print("preprocess called ")
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('ucf_labels.txt'):
                with open('ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        # print(self.fnames[index])
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        
        print("path: ", self.root_dir)

        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        
        print("self.output: ", self.output_dir)

        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    print("&"*100)
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):

        
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count<16:
            return
        
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


def classification_dataset_str_from_arch(arch):
    
    # print("arch: ", arch)

    if 'cifar' in arch:
        dataset = 'cifar10' 
    elif 'mnist' in arch:
        dataset = 'mnist' 
    elif 'slowfast_ucf101' in arch:
        dataset = 'slowfast_ucf101'
    elif 'ucf101' in arch:
        dataset = 'ucf101'
    else:
        dataset = 'imagenet'
    
    # print("dataset: ", dataset)

    return dataset


def classification_num_classes(dataset):
    return {'cifar10': 10,
            'mnist': 10,
            'imagenet': 1000,
            'ucf101': 101,
            'slowfast_ucf101': 101}.get(dataset, None)


def classification_get_input_shape(dataset):
    # print("dataset_class: ", dataset)
    if dataset == 'imagenet':
        return 1, 3, 224, 224
    elif dataset == 'cifar10':
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    elif dataset == 'ucf101':
        return 1, 3, 16, 112, 112
    elif dataset == 'slowfast_ucf101':
        return 1, 3, 16, 112, 112
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return {'cifar10': cifar10_get_datasets,
            'mnist': mnist_get_datasets,
            'imagenet': imagenet_get_datasets}.get(dataset, None)


def load_data(dataset, data_dir, batch_size, workers, validation_split=0.1, deterministic=False,
              effective_train_size=1., effective_valid_size=1., effective_test_size=1.,
              fixed_subset=False, sequential=False, test_only=False):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the dataset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        validation_split: portion of training dataset to set aside for validation
        deterministic: set to True if you want the data loading process to be deterministic.
          Note that deterministic data loading suffers from poor performance.
        effective_train/valid/test_size: portion of the datasets to load on each epoch.
          The subset is chosen randomly each time. For the training and validation sets,
          this is applied AFTER the split to those sets according to the validation_split parameter
        fixed_subset: set to True to keep the same subset of data throughout the run
          (the size of the subset is still determined according to the effective_train/valid/test
          size args)
    """
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')
    
    if(dataset == 'ucf101'):
        train_data = VideoDataset(dataset='ucf101', split='train', clip_len=16, preprocess=False, data_dir=data_dir)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        test_data = VideoDataset(dataset='ucf101', split='test', clip_len=16, preprocess=False, data_dir=data_dir)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        val_data = VideoDataset(dataset='ucf101', split='val', clip_len=16, preprocess=False, data_dir=data_dir)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=workers)

        input_shape = __image_size(test_data)

        return train_loader, val_loader, test_loader, input_shape
    elif dataset == 'slowfast_ucf101':

        train_loader = loader.construct_loader(cfg, "train")
        val_loader = loader.construct_loader(cfg, "val")

        input_shape = __image_size(test_data)

        return train_loader, val_loader,  val_loader, input_shape
        
    else:
        datasets_fn = __dataset_factory(dataset)
        
        return get_data_loaders(datasets_fn, data_dir, batch_size, workers, 
                                validation_split=validation_split,
                                deterministic=deterministic, 
                                effective_train_size=effective_train_size,
                                effective_valid_size=effective_valid_size, 
                                effective_test_size=effective_test_size,
                                fixed_subset=fixed_subset,
                                sequential=sequential,
                                test_only=test_only)


def mnist_get_datasets(data_dir, load_train=True, load_test=True):
    """Load the MNIST dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True,
                                       download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, load_train=True, load_test=True):
    """Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir, load_train=True, load_test=True):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def __image_size(dataset):
    # un-squeeze is used here to add the batch dimension (value=1), which is missing
    return dataset[0][0].unsqueeze(0).size()


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def __split_list(l, ratio):
    split_idx = int(np.floor(ratio * len(l)))
    return l[:split_idx], l[split_idx:]


class SwitchingSubsetRandomSampler(Sampler):
    """Samples a random subset of elements from a data source, without replacement.

    The subset of elements is re-chosen randomly each time the sampler is enumerated

    Args:
        data_source (Dataset): dataset to sample from
        subset_size (float): value in (0..1], representing the portion of dataset to sample at each enumeration.
    """
    def __init__(self, data_source, effective_size):
        self.data_source = data_source
        self.subset_length = _get_subset_length(data_source, effective_size)

    def __iter__(self):
        # Randomizing in the same way as in torch.utils.data.sampler.SubsetRandomSampler to maintain
        # reproducibility with the previous data loaders implementation
        indices = torch.randperm(len(self.data_source))
        subset_indices = indices[:self.subset_length]
        return (self.data_source[i] for i in subset_indices)

    def __len__(self):
        return self.subset_length


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Sequentially samples a subset of the dataset, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def _get_subset_length(data_source, effective_size):
    if effective_size <= 0 or effective_size > 1:
        raise ValueError('effective_size must be in (0..1]')
    return int(np.floor(len(data_source) * effective_size))


def _get_sampler(data_source, effective_size, fixed_subset=False, sequential=False):
    if fixed_subset:
        subset_length = _get_subset_length(data_source, effective_size)
        indices = np.random.permutation(len(data_source))
        subset_indices = indices[:subset_length]
        if sequential:
            return SubsetSequentialSampler(subset_indices)
        else:
            return torch.utils.data.SubsetRandomSampler(subset_indices)
    return SwitchingSubsetRandomSampler(data_source, effective_size)


def get_data_loaders(datasets_fn, data_dir, batch_size, num_workers, validation_split=0.1, deterministic=False,
                     effective_train_size=1., effective_valid_size=1., effective_test_size=1., fixed_subset=False,
                     sequential=False, test_only=False):
    train_dataset, test_dataset = datasets_fn(data_dir, load_train=not test_only, load_test=True)

    worker_init_fn = None
    if deterministic:
        distiller.set_deterministic()
        worker_init_fn = __deterministic_worker_init_fn

    test_indices = list(range(len(test_dataset)))
    test_sampler = _get_sampler(test_indices, effective_test_size, fixed_subset, sequential)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, sampler=test_sampler,
                                              num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(test_dataset)

    # print("input_shape: ", input_shape)

    if test_only:
        return None, None, test_loader, input_shape

    num_train = len(train_dataset)
    indices = list(range(num_train))

    # TODO: Switch to torch.utils.data.datasets.random_split()

    # We shuffle indices here in case the data is arranged by class, in which case we'd would get mutually
    # exclusive datasets if we didn't shuffle
    np.random.shuffle(indices)

    valid_indices, train_indices = __split_list(indices, validation_split)

    train_sampler = _get_sampler(train_indices, effective_train_size, fixed_subset, sequential)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if valid_indices:
        valid_sampler = _get_sampler(valid_indices, effective_valid_size, fixed_subset, sequential)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape
