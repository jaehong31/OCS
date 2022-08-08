import pdb
import numpy as np
import random
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from torch.utils.data.sampler import Sampler, RandomSampler
import torchvision.transforms.functional as TorchVisionFunc
from torchvision.datasets import MNIST
import copy
import os
from core.mixture_loader.mixture import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# PER_TASK_ROATATION = 9
"""
pixel_permutation = [torch.randperm(28*28) for _ in range(20)]
"""
permute_map = {k:np.random.RandomState().permutation(784) for k in range(2, 51)}
permute_map[1] = np.array(range(784))

class Coreset(torch.utils.data.Dataset):
    def __init__(self, set_size, input_shape=[784]):#, task_id=None):
        data_shape = [set_size]+input_shape

        self.data = torch.zeros(data_shape)
        self.targets = torch.ones((set_size))*-1
        #self.task_id = task_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y


def fast_mnist_loader(loaders, eval=True, device='cpu'):
    trains, evals = [], []
    if eval:
        train_loader, eval_loader = loaders
        for data, target in train_loader:
            data = data.to(device).view(-1, 784)
            target = target.to(device)
            trains.append([data, target, None])

        for data, target in eval_loader:
            data = data.to(device).view(-1, 784)
            target = target.to(device)
            evals.append([data, target, None])
        return trains, evals
    else:
        train_loader = loaders

        for data, target in train_loader:
            data = data.to(device).view(-1, 784)
            target = target.to(device)
            trains.append([data, target, None])
        return trains

def fast_cifar_loader(loaders, task_id, eval=True, device='cpu'):
    trains, evals = [], []
    if eval:
        train_loader, eval_loader = loaders
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            trains.append([data, target, task_id])

        for data, target in eval_loader:
            data = data.to(device)
            target = target.to(device)
            evals.append([data, target, task_id])
        return trains, evals
    else:
        for data, target in loaders:
            data = data.to(device)
            target = target.to(device)
            trains.append([data, target, task_id])
        return trains


class RotationTransform:
    """
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TorchVisionFunc.rotate(x, self.angle, fill=(0,))

#########################################################################################################
###        Rotated MNIST
#########################################################################################################
def get_rotated_mnist(task_id, batch_size, per_task_rotation):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # per_task_rotation = PER_TASK_ROATATION
    print('Rotated MNIST')
    rotation_degree = (task_id - 1)*per_task_rotation

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    train_loader = torch.utils.data.DataLoader(MNIST('./data/', train=True, download=True, transform=transforms), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader

def get_subset_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
    # per_task_rotation = PER_TASK_ROATATION

    trains = []
    tests = []
    for i in [task_id]:
        rotation_degree = (i - 1)*per_task_rotation
        transforms = torchvision.transforms.Compose([
                            RotationTransform(rotation_degree),
                            torchvision.transforms.ToTensor(),
        ])
        train = MNIST('./data/', train=True, download=True, transform=transforms)
        test = MNIST('./data/', train=False, download=True, transform=transforms)

        trains.append(train)
        tests.append(test)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def get_multitask_rotated_mnist(num_tasks, batch_size, num_examples, per_task_rotation):
    num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}
    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_mnist_loader(get_subset_rotated_mnist(task, batch_size, num_examples_per_task, per_task_rotation))
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]

        #print('task %d:  %s'%(task, [aa[0].shape for aa in all_mtl_data[task]['train']]))
    return all_mtl_data



#########################################################################################################
###        Permuted MNIST
#########################################################################################################
def get_permuted_mnist(task_id, batch_size):
    idx_permute = permute_map[task_id]
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
                ])
    mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_subset_permuted_mnist(task_id, batch_size, num_examples):
    trains = []
    tests = []
    for i in [task_id]:
        idx_permute = permute_map[task_id]
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
                ])

        train = MNIST('./data/', train=True, download=True, transform=transforms)
        test = MNIST('./data/', train=False, download=True, transform=transforms)

        trains.append(train)
        tests.append(test)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, test_loader

def get_multitask_permuted_mnist(num_tasks, batch_size, num_examples):
    num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}

    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_mnist_loader(get_subset_permuted_mnist(task, batch_size, num_examples_per_task))
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data

#########################################################################################################
###        Split CIFAR
#########################################################################################################
def get_split_cifar100(task_id, batch_size, cifar_train, cifar_test):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """

    start_class = (task_id-1)*5
    end_class = task_id * 5

    targets_train = torch.tensor(cifar_train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    targets_test = torch.tensor(cifar_test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))

    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0]), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def get_subset_split_cifar100(task_id, batch_size, cifar_train, num_examples):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """

    start_class = (task_id-1)*5
    end_class = task_id * 5

    per_class_examples = num_examples//5

    targets_train = torch.tensor(cifar_train.targets)

    # target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
    # train_dataset = torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0])

    trains = []
    for class_number in range(start_class, end_class):
        target = (targets_train == class_number)
        class_train_idx = np.random.choice(np.where(target == 1)[0], per_class_examples, False)
        current_class_train_dataset = torch.utils.data.dataset.Subset(cifar_train, class_train_idx)
        trains.append(current_class_train_dataset)

    trains = ConcatDataset(trains)
    train_loader = torch.utils.data.DataLoader(trains, batch_size=batch_size, shuffle=True)

    return train_loader, []

def get_multitask_cifar100_loaders(num_tasks, batch_size, num_examples):
    num_examples_per_task = num_examples//num_tasks
    trains = []
    tests = []
    all_mtl_data = {}
    cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)

    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_cifar_loader(get_subset_split_cifar100(task, batch_size, cifar_train, num_examples_per_task), task)
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))#trains[:]
        all_mtl_data[task]['val'] = tests[:]
    return all_mtl_data

#########################################################################################################
###        Imbalanced Rotated MNIST
#########################################################################################################
def get_imbalanced_rotated_mnist(task_id, batch_size, per_task_rotation):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # per_task_rotation = PER_TASK_ROATATION
    rotation_degree = (task_id - 1)*per_task_rotation

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    train_dataset = MNIST('./data/', train=True, download=True, transform=transforms)
    test_dataset = MNIST('./data/', train=False, download=True, transform=transforms)

    full_train_x = train_dataset.data
    full_train_y = train_dataset.targets

    # [0,1,2,3,4,5,6,7,8,9]
    len_per_class = []
    idx_per_class = []
    imbalanced_idx = []
    for i in range(10):
        cid_index = torch.where(full_train_y==i, torch.ones_like(full_train_y), torch.zeros_like(full_train_y))
        n_inst = torch.sum(cid_index).item()

        if not (i == 0 or i == 5):
            idx_per_class.append(cid_index.nonzero()[:(int(n_inst/100)*10)])
            len_per_class.append(int(n_inst/100)*10)
        else:
            idx_per_class.append(cid_index.nonzero()[:int(n_inst/10)*10])
            len_per_class.append(int(n_inst/10)*10)

    imbalanced_idx = torch.squeeze(torch.cat(idx_per_class))
    shuffle = torch.randperm(len(imbalanced_idx))

    imbalanced_idx = imbalanced_idx[shuffle]
    train_dataset.data = train_dataset.data[imbalanced_idx]
    train_dataset.targets = train_dataset.targets[imbalanced_idx]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    print('imbalanced Rotated MNIST')
    print('Number of instance per class: %s'%len_per_class)

    return train_loader, test_loader

def get_subset_imbalanced_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
    # per_task_rotation = PER_TASK_ROATATION

    trains = []
    tests = []
    for i in [task_id]:
        rotation_degree = (i - 1)*per_task_rotation
        transforms = torchvision.transforms.Compose([
                            RotationTransform(rotation_degree),
                            torchvision.transforms.ToTensor(),
        ])
        train_dataset = MNIST('./data/', train=True, download=True, transform=transforms)
        test_dataset = MNIST('./data/', train=False, download=True, transform=transforms)

        full_train_x = train_dataset.data
        full_train_y = train_dataset.targets

        # [0,1,2,3,4,5,6,7,8,9]
        len_per_class = []
        idx_per_class = []
        imbalanced_idx = []
        for ii in range(10):
            cid_index = torch.where(full_train_y==ii, torch.ones_like(full_train_y), torch.zeros_like(full_train_y))
            n_inst = torch.sum(cid_index).item()

            if not (ii == 0 or ii == 5):
                idx_per_class.append(cid_index.nonzero()[:(int(n_inst/100)*10)])
                len_per_class.append(int(n_inst/100)*10)
            else:
                idx_per_class.append(cid_index.nonzero()[:int(n_inst/10)*10])
                len_per_class.append(int(n_inst/10)*10)

        imbalanced_idx = torch.squeeze(torch.cat(idx_per_class))
        shuffle = torch.randperm(len(imbalanced_idx))

        imbalanced_idx = imbalanced_idx[shuffle]
        train_dataset.data = train_dataset.data[imbalanced_idx]
        train_dataset.targets = train_dataset.targets[imbalanced_idx]

        trains.append(train_dataset)
        tests.append(test_dataset)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)
    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def get_multitask_imbalanced_rotated_mnist(num_tasks, batch_size, num_examples, per_task_rotation):
    num_examples_per_task = num_examples//num_tasks

    trains = []
    tests = []
    all_mtl_data = {}
    for task in range(1, num_tasks+1):
        all_mtl_data[task] = {}
        train_loader, test_loader = fast_mnist_loader(get_subset_imbalanced_rotated_mnist(task, batch_size, num_examples_per_task, per_task_rotation))
        trains += train_loader
        tests += test_loader
        all_mtl_data[task]['train'] = random.sample(trains[:], len(trains))
        all_mtl_data[task]['val'] = tests[:]

        #print('task %d:  %s'%(task, [aa[0].shape for aa in all_mtl_data[task]['train']]))
    return all_mtl_data


#########################################################################################################
###        Imbalanced SplitCifar100
#########################################################################################################
def get_imbalanced_split_cifar100(task_id, batch_size, cifar_train, cifar_test):

    class_shuffle = [25, 11, 23, 76, 12, 30, 62,  6, 89, 44, 84, 29, 82,  3, 10, 24, 64, 72, \
        21,  8, 63, 71, 68, 74,  5, 86, 22, 58, 95, 19, 47, 54, 56,  2, 20, 96, \
        57, 38, 80, 66,  1, 59, 16, 97, 18, 73, 31, 77, 99, 15, 46, 27, 83, 40, \
        48, 88, 33, 36, 81, 55, 85, 14, 13,  7, 65, 50, 78, 43, 91,  4, 69, 52, \
        41, 94, 34, 51, 37,  9, 90, 35, 92, 26, 42,  0, 17, 87, 53, 93, 32, 28, \
        75, 67, 49, 79, 61, 60, 70, 98, 45, 39]


    n_class = 100
    full_train_x = cifar_train.data
    full_train_y = cifar_train.targets

    start_class = (task_id-1)*5
    end_class = task_id * 5

    len_per_class = []
    idx_per_class = []
    imbalanced_idx = []
    img_max = len(cifar_train.targets) // n_class
    imb_factor = 1/10.
    for class_number in range(n_class):
        cid_index = np.asarray([i for i, c in enumerate(cifar_train.targets) if c == class_shuffle[class_number]])
        num_sample = int(img_max * (imb_factor**(class_number/(n_class - 1))))
        idx_per_class.append(torch.from_numpy(cid_index[:num_sample]))
        len_per_class.append(num_sample)


    cifar_train_new = copy.deepcopy(cifar_train)

    imbalanced_idx = torch.squeeze(torch.cat(idx_per_class))
    shuffle = torch.randperm(len(imbalanced_idx))

    imbalanced_idx = imbalanced_idx[shuffle]
    imbalanced_idx = imbalanced_idx.numpy()
    cifar_train_new.data = cifar_train.data[imbalanced_idx]
    cifar_train_new.targets = [cifar_train.targets[i] for i in imbalanced_idx]

    target_train_idx = [1 if ((i >= start_class) & (i < end_class)) else 0 for i in cifar_train_new.targets]
    target_test_idx = [1 if ((i >= start_class) & (i < end_class)) else 0 for i in cifar_test.targets]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_train_new, np.where(np.array(target_train_idx)==1)[0]), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(np.array(target_test_idx)==1)[0]), batch_size=batch_size, shuffle=True)

    print('imbalanced SplitCifar100')
    if task_id == 1:
        print('Number of instance per class: %s'%len_per_class)
    pppp = np.array(cifar_train_new.targets)[np.where(np.array(target_train_idx)==1)[0]]
    print('Number of instance per class: for task %d %s'%(task_id, [sum(pppp == se) for se in range(start_class, end_class)]))

    return train_loader, test_loader

def get_subset_imbalanced_split_cifar100(task_id, batch_size, cifar_train, num_examples):
    # per_task_rotation = PER_TASK_ROATATION

    start_class = (task_id-1)*5
    end_class = task_id * 5
    n_class = 100
    targets_train = torch.tensor(cifar_train.targets)

    img_max = len(cifar_train.targets) // n_class
    imb_factor = 1. / 100
    trains = []
    for class_number in range(start_class, end_class):
        target = (targets_train == class_number)
        per_class_examples = int(img_max * (imb_factor**(class_number/(n_class - 1))))
        class_train_idx = np.random.choice(np.where(target == 1)[0], per_class_examples, False)
        current_class_train_dataset = torch.utils.data.dataset.Subset(cifar_train, class_train_idx)
        trains.append(current_class_train_dataset)

    trains = ConcatDataset(trains)
    train_loader = torch.utils.data.DataLoader(trains, batch_size=batch_size, shuffle=True)

    return train_loader, []


#########################################################################################################
###        Rotated Balanced Noisy MNIST
#########################################################################################################

def get_balanced_noisy_rotated_mnist(task_id, batch_size, per_task_rotation):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # per_task_rotation = PER_TASK_ROATATION
    print('Noisy Rotated MNIST')
    rotation_degree = (task_id - 1)*per_task_rotation

    train_transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    test_transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])

    train_dataset = MNIST('./data/', train=True, download=True, transform=train_transforms)
    test_dataset = MNIST('./data/', train=False, download=True, transform=test_transforms)
    end_idx = int(len(train_dataset.data) * 0.6)
    shuffle = torch.randperm(len(train_dataset.data))
    idx = shuffle[:end_idx].long()
    for i in idx:
        train_dataset.data[i] = torch.randn(train_dataset.data[i].size())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(MNIST('./data/', train=False, download=True, transform=test_transforms),  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader


def get_subset_balanced_noisy_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
    # per_task_rotation = PER_TASK_ROATATION

    trains = []
    tests = []
    for i in [task_id]:
        rotation_degree = (i - 1) * per_task_rotation
        train_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor(),
                ])
        test_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor()
                ])

        train = MNIST('./data/', train=True, download=True, transform=train_transforms)
        test = MNIST('./data/', train=False, download=True, transform=test_transforms)

        trains.append(train)
        tests.append(test)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)

    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, test_loader


#########################################################################################################
###        Imbalanced Rotated Noisy MNIST
#########################################################################################################
def get_imbalanced_rotated_mnist(task_id, batch_size, per_task_rotation):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    # per_task_rotation = PER_TASK_ROATATION
    rotation_degree = (task_id - 1)*per_task_rotation

    train_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor(),
                ])
    test_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor()
                ])

    train_dataset = MNIST('./data/', train=True, download=True, transform=train_transforms)
    test_dataset = MNIST('./data/', train=False, download=True, transform=test_transforms)

    full_train_x = train_dataset.data
    full_train_y = train_dataset.targets

    # [0,1,2,3,4,5,6,7,8,9]
    len_per_class = []
    idx_per_class = []
    imbalanced_idx = []
    for i in range(10):
        cid_index = torch.where(full_train_y==i, torch.ones_like(full_train_y), torch.zeros_like(full_train_y))
        n_inst = torch.sum(cid_index).item()

        if not (i == 0 or i == 5):
            idx_per_class.append(cid_index.nonzero()[:(int(n_inst/100)*10)])
            len_per_class.append(int(n_inst/100)*10)
        else:
            idx_per_class.append(cid_index.nonzero()[:int(n_inst/10)*10])
            len_per_class.append(int(n_inst/10)*10)

    imbalanced_idx = torch.squeeze(torch.cat(idx_per_class))
    shuffle = torch.randperm(len(imbalanced_idx))

    imbalanced_idx = imbalanced_idx[shuffle]
    train_dataset.data = train_dataset.data[imbalanced_idx]
    train_dataset.targets = train_dataset.targets[imbalanced_idx]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,  batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    print('imbalanced Rotated MNIST')
    print('Number of instance per class: %s'%len_per_class)

    return train_loader, test_loader

def get_subset_imbalanced_rotated_mnist(task_id, batch_size, num_examples, per_task_rotation):
    # per_task_rotation = PER_TASK_ROATATION

    trains = []
    tests = []
    for i in [task_id]:
        rotation_degree = (i - 1)*per_task_rotation
        train_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor(),
                ])
        test_transforms = torchvision.transforms.Compose([
                RotationTransform(rotation_degree),
                torchvision.transforms.ToTensor()
                ])

        train_dataset = MNIST('./data/', train=True, download=True, transform=train_transforms)
        test_dataset = MNIST('./data/', train=False, download=True, transform=test_transforms)

        full_train_x = train_dataset.data
        full_train_y = test_dataset.targets

        # [0,1,2,3,4,5,6,7,8,9]
        len_per_class = []
        idx_per_class = []
        imbalanced_idx = []
        for ii in range(10):
            cid_index = torch.where(full_train_y==ii, torch.ones_like(full_train_y), torch.zeros_like(full_train_y))
            n_inst = torch.sum(cid_index).item()

            if not (ii == 0 or ii == 5):
                idx_per_class.append(cid_index.nonzero()[:(int(n_inst/100)*10)])
                len_per_class.append(int(n_inst/100)*10)
            else:
                idx_per_class.append(cid_index.nonzero()[:int(n_inst/10)*10])
                len_per_class.append(int(n_inst/10)*10)

        imbalanced_idx = torch.squeeze(torch.cat(idx_per_class))
        shuffle = torch.randperm(len(imbalanced_idx))

        imbalanced_idx = imbalanced_idx[shuffle]
        train_dataset.data = train_dataset.data[imbalanced_idx]
        train_dataset.targets = train_dataset.targets[imbalanced_idx]

        trains.append(train_dataset)
        tests.append(test_dataset)

    train_datasets = ConcatDataset(trains)
    test_datasets = ConcatDataset(tests)

    # num_examples = num_examples_per_task * num_tasks
    sampler = torch.utils.data.RandomSampler(train_datasets, replacement=True, num_samples=num_examples)
    train_loader = torch.utils.data.DataLoader(train_datasets,  batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_datasets,  batch_size=256, shuffle=True, num_workers=0, pin_memory=True)
    return train_loader, test_loader



#########################################################################################################
###        Loader
#########################################################################################################

def get_all_loaders(dataset, num_tasks, bs_inter, bs_intra, num_examples, per_task_rotation=9, is_coreset=True):
    dataset = dataset.lower()
    loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'coreset': {}, 'full-multitask': {}}
    print('loading multitask {}'.format(dataset))
    class_arr = range(0, 100)
    if 'cifar' in dataset:
        loaders['multitask'] = get_multitask_cifar100_loaders(num_tasks, bs_inter, num_examples)
        cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
        cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
        cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
    elif 'rot' in dataset and 'mnist' in dataset:
        if 'imb' in dataset:
            loaders['multitask'] = get_multitask_imbalanced_rotated_mnist(num_tasks, bs_inter, num_examples, per_task_rotation)
        elif 'noisy' in dataset:
            pass
        else:
            loaders['multitask'] = get_multitask_rotated_mnist(num_tasks, bs_inter, num_examples, per_task_rotation)
    elif 'perm' in dataset and 'mnist' in dataset:
        loaders['multitask'] = get_multitask_permuted_mnist(num_tasks, bs_inter, num_examples)
    else:
        raise Exception("{} not implemented!".format(dataset))

    if is_coreset:
        print('loading coreset placeholder {}'.format(dataset))
        for task in range(1, num_tasks+1):
            loaders['sequential'][task], loaders['coreset'][task], loaders['subset'][task] = {}, {}, {}
            print("loading {} for task {}".format(dataset, task))
            c_size = num_examples//task
            classes = np.random.choice(class_arr, 5, replace=False)
            if 'rot' in dataset and 'mnist' in dataset:
                if 'imb' in dataset:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_imbalanced_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_imbalanced_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
                elif 'noisy' in dataset:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_balanced_noisy_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_balanced_noisy_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
                else:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
                loaders['coreset'][task]['train'] = Coreset(c_size)
            elif 'cifar' in dataset:
                if 'imb' in dataset:
                    seq_loader_train , seq_loader_val = fast_cifar_loader(get_imbalanced_split_cifar100(task, bs_intra, cifar_train, cifar_test), task, 'cpu')
                    sub_loader_train , _ = fast_cifar_loader(get_subset_imbalanced_split_cifar100(task, bs_inter, cifar_train, 5*num_examples), task, 'cpu')
                else:
                    seq_loader_train , seq_loader_val = fast_cifar_loader(get_split_cifar100(task, bs_intra, cifar_train, cifar_test), task, 'cpu')
                    sub_loader_train , _ = fast_cifar_loader(get_subset_split_cifar100(task, bs_inter, cifar_train, 5*num_examples), task, 'cpu')
                loaders['coreset'][task]['train'] = Coreset(c_size, [3, 32, 32])
            loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
            loaders['subset'][task]['train'] = sub_loader_train
        return loaders
    else:
        # Load sequential tasks
        for task in range(1, num_tasks+1):
            loaders['sequential'][task], loaders['subset'][task] = {}, {}
            print("loading {} for task {}".format(dataset, task))
            if 'rot' in dataset and 'mnist' in dataset:
                if 'imb' in dataset:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_imbalanced_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_imbalanced_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
                elif 'noisy' in dataset:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_balanced_noisy_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_balanced_noisy_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
                else:
                    seq_loader_train , seq_loader_val = fast_mnist_loader(get_rotated_mnist(task, bs_intra, per_task_rotation), 'cpu')
                    sub_loader_train , _ = fast_mnist_loader(get_subset_rotated_mnist(task, bs_inter, num_examples, per_task_rotation),'cpu')
            elif 'cifar' in dataset:
                seq_loader_train , seq_loader_val = fast_cifar_loader(get_split_cifar100(task, bs_intra, cifar_train, cifar_test), task, 'cpu')
                sub_loader_train , _ = fast_cifar_loader(get_subset_split_cifar100(task, bs_inter, cifar_train, 5*num_examples), task, 'cpu')
            loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
            loaders['subset'][task]['train'] = sub_loader_train
        return loaders


#########################################################################################################
###        Lixture Loader
#########################################################################################################


def get_mixture_loader(task_id, batch_size, dataset, ncla, shuffle=True, full_train=True, train_size=1000, imb_idx=None, device='cpu'):
    start_class = sum(ncla[:task_id-1]) if task_id > 1 else 0
    end_class = sum(ncla[:task_id])

    dataset['train']['y'] += start_class
    dataset['test']['y'] += start_class

    trains, evals = [], []
    total_train = len(dataset['train']['y'])
    if shuffle:
        shuf = torch.randperm(total_train)
        dataset['train']['x'] = dataset['train']['x'][shuf]
        dataset['train']['y'] = dataset['train']['y'][shuf]

    # imbalanced
    if isinstance(imb_idx, list) and ncla[task_id-1] == 10:
        idx_per_class = []
        for class_number in range(start_class, end_class):
            cid_index = np.asarray([i for i, c in enumerate(dataset['train']['y']) if c == class_number])
            idx_per_class.append(torch.from_numpy(cid_index[:imb_idx[class_number]]))

        reduced = torch.cat(idx_per_class)
        _shuffle = torch.randperm(len(reduced))

        dataset['train']['x'] = dataset['train']['x'][reduced[_shuffle]]
        dataset['train']['y'] = dataset['train']['y'][reduced[_shuffle]]

    iaa = []
    for class_number in range(start_class, end_class):
        cid_index = np.asarray([i for i, c in enumerate(dataset['train']['y']) if c == class_number])
        iaa.append(len(cid_index))
    print(iaa)

    if full_train:
        iteration = round(total_train / batch_size)
    else:
        t_size = min(len(dataset['train']['x']), train_size)
        iteration = round(t_size / batch_size)
    index = 0
    for i in range(iteration):
        offset = min(batch_size, total_train-index)
        data = dataset['train']['x'][index:index+offset].to(device)
        target = dataset['train']['y'][index:index+offset].to(device)
        trains.append([data, target, task_id])
        index += batch_size

    total_test = len(dataset['test']['y'])
    iteration = round(total_test / batch_size)
    index = 0
    for i in range(iteration):
        offset = min(batch_size, total_test-index)
        data = dataset['test']['x'][index:index+offset].to(device)
        target = dataset['test']['y'][index:index+offset].to(device)
        evals.append([data, target, task_id])
        index += batch_size

    return trains, evals

class MixtureDataset:
    def __init__(self, opt, valid=False, is_imbalanced=False):
        self.is_imb = is_imbalanced
        self.opt = opt
        if valid:
            self.seprate_ratio = (0.7, 0.2, 0.1) # train, test, valid
        else:
            self.seprate_ratio = (0.7, 0.3)
        self.mixture_dir = './data/mixture/'
        self.mixture_filename = 'mixture2.npy'
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'traffic_sign',
            6: 'face_scrub',
            7: 'not_mnist',
        }

    def get_imbalanced_idx(self, n_class=40, img_max=int(1000/10)):
        class_shuffle = [31,  9, 11, 39, 32, 34, 13,  5, 20, 29,  1, 36, 24, 14, 25, 26,  7, 12, \
                            23, 18, 10, 38,  8, 17, 27,  3, 35,  2, 28,  0, 21,  6, 37, 15, 33, 22, 19, 16, 30,  4]

        len_per_class = []
        imbalanced_idx = []
        imb_factor = 1/10.
        for class_number in range(n_class):
            num_sample = int(img_max * (imb_factor**(class_number/(n_class - 1))))
            len_per_class.append(num_sample)
        out =  np.array(len_per_class)[class_shuffle].tolist()
        oqq = out[:30] + [0 for _ in range(43)] + out[30:]
        return oqq

    def get_loader(self):
        return self.generate_data()

    def generate_data(self):
        saved_mixture_filepath = os.path.join(self.mixture_dir, self.mixture_filename)
        if os.path.exists(saved_mixture_filepath):
            print('loading mixture data: {}'.format(saved_mixture_filepath))
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            mixture = get(base_dir=self.mixture_dir, pc_valid=0.0, fixed_order=True)
        ncla = [mixture[0][tid]['ncla'] for tid in range(5)]
        name = [mixture[0][tid]['name'] for tid in range(5)]

        if self.is_imb:
            imb_idx = self.get_imbalanced_idx()
            loader = self.get_loaders(mixture[0], ncla, name, imb_idx)
        else:
            loader = self.get_loaders(mixture[0], ncla, name)
        return loader


    def get_loaders(self, mixture, ncla, name, imb_idx=None):
        loaders = {'sequential': {},  'multitask':  {}, 'subset': {}, 'coreset': {}, 'full-multitask': {}}
        print('loading coreset placeholder MixtureDataset')
        for task in range(1, self.opt['num_tasks']+1):
            loaders['sequential'][task], loaders['coreset'][task], loaders['subset'][task] = {}, {}, {}
            print("loading {} for task {}".format(name[task-1], task))
            c_size = ncla[task-1]
            seq_loader_train , seq_loader_val = get_mixture_loader(task, self.opt['stream_size'], mixture[task-1], ncla, full_train=False, imb_idx=imb_idx)
            loaders['coreset'][task]['train'] = Coreset(c_size, [3, 32, 32])
            loaders['sequential'][task]['train'], loaders['sequential'][task]['val'] = seq_loader_train, seq_loader_val
        return loaders
