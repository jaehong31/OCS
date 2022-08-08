import torch
import pdb
import random
import copy
import torch.nn as nn
import numpy as np
from .utils import DEVICE, load_model, assign_weights, flatten_params
from .data_utils import Coreset, fast_mnist_loader, fast_cifar_loader

def get_line_loss(start_w, w, loader, config, is_prev=False, task=0):
    interpolation  = None
    if 'line' in config['lmc_interpolation'] or 'integral' in config['lmc_interpolation']:
        interpolation = 'linear'
    elif 'stochastic' in config['lmc_interpolation']:
        interpolation = 'stochastic'
    else:
        raise Exception("non-implemented interpolation")

    m = load_model('{}/{}.pth'.format(config['exp_dir'], 'init')).to(DEVICE)
    total_loss = 0
    accum_grad = None
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    if interpolation == 'linear':
        for t in np.arange(0.0, 1.01, 1.0/float(config['lmc_line_samples'])):
            grads = []
            cur_weight = start_w + (w - start_w) * t
            m = assign_weights(m, cur_weight).to(DEVICE)

            if config['is_coreset'] and is_prev:
                current_loss = get_rcp_clf_loss(m, loader, config, task)
            else:
                current_loss = get_clf_loss(m, loader)
            current_loss.backward()

            for name, param in m.named_parameters():
                grads.append(param.grad.view(-1))
            grads = torch.cat(grads)
            if accum_grad is None:
                accum_grad = grads
            else:
                accum_grad += grads
        return accum_grad
    elif interpolation == 'stochastic':
        for data, target, task_id in loader:
                grads = []
                t = np.random.uniform()
                cur_weight = start_w + (w - start_w) * t
                m = assign_weights(m, cur_weight).to(DEVICE)
                m.eval()
                data = data.to(DEVICE)
                target = target.to(DEVICE)
                output = m(data, task_id)
                current_loss = criterion(output, target)
                current_loss.backward()
                for name, param in m.named_parameters():
                    grads.append(param.grad.view(-1))
                grads = torch.cat(grads)
                if accum_grad is None:
                    accum_grad = grads
                else:
                    accum_grad += grads
        return accum_grad

    else:
        return None

def get_clf_loss(net, loader):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.eval()
    test_loss = 0
    count = 0
    for data, target, task_id in loader:
            count += len(target)
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = net(data, task_id)
            test_loss += criterion(output, target)
    test_loss /= count
    return test_loss

def reconstruct_coreset_loader2(config, dataset, task):
    trains = []
    all_coreset = {}

    for tid in range(1,task+1):
        if 'mixture' in config['dataset']:
            num_examples_per_task = config['n_classes'][tid]
        else:
            num_examples_per_task = config['memory_size'] // task
        coreset = Coreset(num_examples_per_task)

        pick_idx = torch.randperm(num_examples_per_task)
        coreset.data = copy.deepcopy(dataset[tid]['train'].data[pick_idx])
        coreset.targets = copy.deepcopy(dataset[tid]['train'].targets[pick_idx])
        coreset_loader = torch.utils.data.DataLoader(coreset, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        if ('mixture' in config['dataset']) or ('cifar' in config['dataset']):
            train_loader = fast_cifar_loader(coreset_loader, tid, eval=False)
        else:
            train_loader = fast_mnist_loader(coreset_loader, eval=False)

        trains += train_loader
    all_coreset = random.sample(trains[:], len(trains))
    return all_coreset


def get_rcp_clf_loss(net, dataset, config, task):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.eval()
    test_loss = 0
    count = 0
    loader = reconstruct_coreset_loader2(config, dataset, task)

    for data, target, task_id in loader:
            count += len(target)
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = net(data, task_id)
            test_loss += criterion(output, target.long())
    test_loss /= count
    return test_loss

def get_coreset_loss(net, iterloader, config):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    net.train()
    coreset_loss = 0
    count = 0
    data, target, task_id = iterloader
    count += len(target)
    data = data.to(DEVICE)
    target = target.to(DEVICE)
    output = net(data, task_id)
    coreset_loss += criterion(output, target.long())
    coreset_loss /= count
    return coreset_loss

def get_ti(input, labels=None, dataset='cifar'):
    # get task identity
    ll = labels
    if dataset == 'cifar':
        for lid in range(len(ll)):
            t = np.int(labels[lid].item() / 5)
            offset1 = int(t * 5)
            offset2 = int((t + 1) * 5)
            if offset1 > 0:
                input[lid, :offset1].data.fill_(-10e10)
            if offset2 < 100:
                input[lid, offset2:100].data.fill_(-10e10)
    else:
        pass
    return input
