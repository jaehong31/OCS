import pdb
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributions as td

from .utils import DEVICE, save_model,load_model
from .utils import flatten_params, flatten_grads, flatten_example_grads, assign_weights, assign_grads, accum_grads
from .mode_connectivity import get_line_loss, get_coreset_loss, reconstruct_coreset_loader2
from .summary import Summarizer
from .bilevel_coreset import BilevelCoreset
from .ntk_generator import generate_fnn_ntk, generate_cnn_ntk

import copy
import core.autograd_hacks as autograd_hacks
gumbel_dist = td.gumbel.Gumbel(0,1)


coreset_methods = ['uniform', 'coreset',
           'kmeans_features', 'kcenter_features', 'kmeans_grads',
           'kmeans_embedding', 'kcenter_embedding', 'kcenter_grads',
           'entropy', 'hardest', 'frcl', 'icarl', 'grad_matching']

def get_kernel_fn():
    return lambda x, y: generate_cnn_ntk(x.reshape(-1, 28, 28, 1), y.reshape(-1, 28, 28, 1))

def coreset_cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value

kernel_fn = get_kernel_fn()
bc = BilevelCoreset(outer_loss_fn=coreset_cross_entropy,
        inner_loss_fn=coreset_cross_entropy, out_dim=10, max_outer_it=1,
        max_inner_it=200, logging_period=10)

def sample_selection(g, eg, config, ref_grads=None, attn=None):
    ng = torch.norm(g)
    neg = torch.norm(eg, dim=1)
    mean_sim = torch.matmul(g,eg.t()) / torch.maximum(ng*neg, torch.ones_like(neg)*1e-6)
    negd = torch.unsqueeze(neg, 1)

    cross_div = torch.matmul(eg,eg.t()) / torch.maximum(torch.matmul(negd, negd.t()), torch.ones_like(negd)*1e-6)
    avg_div = torch.mean(cross_div, 0)

    coreset_aff = 0.
    if ref_grads is not None:
        ref_ng = torch.norm(ref_grads)
        coreset_aff = torch.matmul(ref_grads, eg.t()) / torch.maximum(ref_ng*neg, torch.ones_like(neg)*1e-6)

    measure = mean_sim - avg_div + config['tau'] * coreset_aff
    _, u_idx = torch.sort(measure, descending=True)
    return u_idx.cpu().numpy()

def classwise_fair_selection(task, cand_target, sorted_index, num_per_label, config, is_shuffle=True):
    num_examples_per_task = config['memory_size'] // task
    num_examples_per_class = num_examples_per_task // config['n_classes']
    num_residuals = num_examples_per_task - num_examples_per_class * config['n_classes']
    residuals =  np.sum([(num_examples_per_class - n_c)*(num_examples_per_class > n_c) for n_c in num_per_label])
    num_residuals += residuals

    # Get the number of coreset instances per class
    while True:
        n_less_sample_class =  np.sum([(num_examples_per_class > n_c) for n_c in num_per_label])
        num_class = (config['n_classes']-n_less_sample_class)
        if (num_residuals // num_class) > 0:
            num_examples_per_class += (num_residuals // num_class)
            num_residuals -= (num_residuals // num_class) * num_class
        else:
            break
    # Get best coresets per class
    selected = []
    for j in range(config['n_classes']):
        position = np.squeeze((cand_target[sorted_index]==j).nonzero())
        if position.numel() > 1:
            selected.append(position[:num_examples_per_class])
        elif position.numel() == 0:
            continue
        else:
            selected.append([position])
    # Fill rest space as best residuals
    selected = np.concatenate(selected)
    unselected = np.array(list(set(np.arange(num_examples_per_task))^set(selected)))
    final_num_residuals = num_examples_per_task - len(selected)
    best_residuals = unselected[:final_num_residuals]
    selected = np.concatenate([selected, best_residuals])

    if is_shuffle:
        random.shuffle(selected)

    return sorted_index[selected.astype(int)]


def select_coreset(loader, task, model, candidates, config, candidate_size=1000, fair_selection=True):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    temp_optimizer = torch.optim.SGD(model.parameters(), lr=config['seq_lr'], momentum=config['momentum'])
    temp_optimizer.zero_grad()

    if fair_selection:
        # collect candidates
        cand_data, cand_target = [], []
        cand_size = len(candidates)
        for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
            if batch_idx == cand_size:
                break
            cand_data.append(data[candidates[batch_idx]])
            cand_target.append(target[candidates[batch_idx]])

        cand_data = torch.cat(cand_data, 0)
        cand_target = torch.cat(cand_target, 0)

        random_pick_up = torch.randperm(len(cand_target))[:candidate_size]
        cand_data = cand_data[random_pick_up]
        cand_target = cand_target[random_pick_up]

        num_per_label = [len((cand_target==jj).nonzero()) for jj in range(config['n_classes'])]
        #print('num samples per label', num_per_label)

        num_examples_per_task = config['memory_size'] // task

        if config['select_type'] in coreset_methods:
            rs = np.random.RandomState(0)
            if config['select_type'] == 'coreset':
                pick, _, = bc.build_with_representer_proxy_batch(cand_data.cpu().numpy(), cand_target.cpu().numpy(), num_examples_per_task, kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=1e-3)
            else:
                summarizer = Summarizer.factory(config['select_type'], rs)
                pick = summarizer.build_summary(cand_data.cpu().numpy(), cand_target.cpu().numpy(), num_examples_per_task, method=config['select_type'], model=model, device=DEVICE)
            loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[pick])
            loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[pick])
        else:
            autograd_hacks.clear_backprops(model)
            pred = model(cand_data.to(DEVICE), task)
            loss = criterion(pred, cand_target.long().to(DEVICE))
            loss.backward()
            autograd_hacks.compute_grad1(model)

            # Coreset update
            _g = flatten_grads(model)
            _eg = flatten_example_grads(model)
            sorted = sample_selection(_g, _eg, config)
            if config['select_type'] == 'ocs_select':
                pick = torch.randperm(len(sorted))
                selected = classwise_fair_selection(task, cand_target, pick, num_per_label, config, is_shuffle=True)
            else:
                pass

            loader['coreset'][task]['train'].data = copy.deepcopy(cand_data[selected])
            loader['coreset'][task]['train'].targets = copy.deepcopy(cand_target[selected])
            num_per_label = [len((cand_target[selected]==jj).nonzero()) for jj in range(config['n_classes'])]
            #print('after select_coreset, num samples per label', num_per_label)
    else:
        pass

def update_coreset(loader, task, model, task_id, config):
    # Coreset update
    num_examples_per_task = config['memory_size'] // task
    prv_nept = config['memory_size'] // (task-1)

    for tid in range(1, task):
        if config['select_type'] in coreset_methods:
            xx = num_examples_per_task if tid == 1 else prv_nept
            tid_coreset = loader['coreset'][tid]['train'].data
            tid_targets = loader['coreset'][tid]['train'].targets
            class_idx = [tid_targets.cpu().numpy() == i for i in range(config['n_classes'])]
            num_per_label = [len((tid_targets.cpu()==jj).nonzero()) for jj in range(config['n_classes'])]
            rs = np.random.RandomState(0)
            if config['select_type'] == 'coreset':
                selected, _, = bc.build_with_representer_proxy_batch(loader['coreset'][tid]['train'].data.cpu().numpy(), loader['coreset'][tid]['train'].targets.cpu().numpy(), num_examples_per_task, kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=1e-3)
            else:
                summarizer = Summarizer.factory(config['select_type'], rs)
                selected = summarizer.build_summary(loader['coreset'][tid]['train'].data.cpu().numpy(), loader['coreset'][tid]['train'].targets.cpu().numpy(), num_examples_per_task, method=config['select_type'], model=model, device=DEVICE)
        elif config['select_type'] == 'ocs_select':
            criterion = nn.CrossEntropyLoss().to(DEVICE)
            temp_optimizer = torch.optim.SGD(model.parameters(), lr=config['seq_lr'], momentum=config['momentum'])

            tid_coreset = loader['coreset'][tid]['train'].data
            tid_targets = loader['coreset'][tid]['train'].targets

            temp_optimizer.zero_grad()
            autograd_hacks.clear_backprops(model)

            pred = model(tid_coreset.to(DEVICE), task_id)
            loss = criterion(pred, tid_targets.long().to(DEVICE))
            loss.backward()
            autograd_hacks.compute_grad1(model)

            _tid_g = flatten_grads(model)
            _tid_eg = flatten_example_grads(model)
            sorted = sample_selection(_tid_g, _tid_eg, config)

            class_idx = [tid_targets.cpu().numpy() == i for i in range(config['n_classes'])]
            num_per_label = [len((tid_targets.cpu()==jj).nonzero()) for jj in range(config['n_classes'])]

            selected = classwise_fair_selection(task, tid_targets, sorted, num_per_label, config)
        _nn = [len((tid_targets[selected]==jj).nonzero()) for jj in range(config['n_classes'])]

        #print('tid:%d, after update_coreset, num samples per label'%tid, _nn)
        loader['coreset'][tid]['train'].data = copy.deepcopy(loader['coreset'][tid]['train'].data[selected])
        loader['coreset'][tid]['train'].targets = copy.deepcopy(loader['coreset'][tid]['train'].targets[selected])


def train_single_step(model, optimizer, loader, task, step, config):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == config['n_substeps'] else False
    rs = np.random.RandomState(0)
    if config['select_type'] in coreset_methods and config['select_type'] != 'coreset':
        summarizer = Summarizer.factory(config['select_type'], rs)

    candidates_indices=[]
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        optimizer.zero_grad()
        is_rand_start = True if ((step == 1) and (batch_idx < config['r2c_iter']) and config['is_r2c']) else False
        is_ocspick = True if (config['ocspick'] and len(data) > config['batch_size']) else False
        if is_ocspick and not is_rand_start:
            autograd_hacks.clear_backprops(model)
            pred = model(data, task_id)
            loss = criterion(pred, target)
            loss.backward()
            autograd_hacks.compute_grad1(model)

            # Coreset update
            _g = flatten_grads(model)
            _eg = flatten_example_grads(model)
            sorted = sample_selection(_g, _eg, config)

            pick = sorted[:config['batch_size']]
            autograd_hacks.clear_backprops(model)
            optimizer.zero_grad()

            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()

            # Select coresets at final step
            if is_last_step:
                candidates_indices.append(pick)

        elif config['select_type'] in coreset_methods:
            size = min(len(data), config['batch_size'])
            if config['select_type'] == 'coreset':
                if is_last_step and batch_idx < 10:
                    selected_pick, _, = bc.build_with_representer_proxy_batch(data.cpu().numpy(), target.cpu().numpy(), config['batch_size'], kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=1e-3)
                else:
                    selected_pick = torch.randperm(len(data))[:size]
            else:
                selected_pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(), config['batch_size'], method=config['select_type'], model=model, device=DEVICE)

            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()
            if is_last_step:
                candidates_indices.append(pick)
        else:
            size = min(len(data), config['batch_size'])
            pick = torch.randperm(len(data))[:size]
            pred = model(data[pick], task_id)
            loss = criterion(pred, target[pick])
            loss.backward()
        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)

    return model

def train_ocs_single_step(model, optimizer, loader, task, step, config):
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    is_last_step = True if step == config['n_substeps'] else False
    prev_coreset = [loader['coreset'][tid]['train'].data for tid in range(1, task)]
    prev_targets = [loader['coreset'][tid]['train'].targets for tid in range(1, task)]
    c_x = torch.cat(prev_coreset, 0)
    c_y = torch.cat(prev_targets, 0)

    ref_loader = reconstruct_coreset_loader2(config, loader['coreset'], task-1)
    ref_iterloader = iter(ref_loader)

    candidates_indices=[]
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        optimizer.zero_grad()
        is_rand_start = True if ((step == 1) and (batch_idx < config['r2c_iter']) and config['is_r2c']) else False
        is_ocspick = True if (config['ocspick'] and len(data) > config['batch_size']) else False

        # Compute reference grads
        ref_pred = model(c_x.to(DEVICE), None)
        ref_loss = criterion(ref_pred, c_y.long().to(DEVICE))
        ref_loss.backward()
        ref_grads = copy.deepcopy(flatten_grads(model))
        optimizer.zero_grad()

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        if is_rand_start:
            size = min(len(data), config['batch_size'])
            pick = torch.randperm(len(data))[:size]
        else:
            autograd_hacks.clear_backprops(model)
            pred = model(data, task_id)
            loss = criterion(pred, target)
            loss.backward()
            autograd_hacks.compute_grad1(model)

            # Coreset update
            _g = flatten_grads(model)
            _eg = flatten_example_grads(model)
            sorted = sample_selection(_g, _eg, config, ref_grads=ref_grads)
            pick = sorted[:config['batch_size']]
            autograd_hacks.clear_backprops(model)
            optimizer.zero_grad()

        pred = model(data[pick], task_id)
        loss = criterion(pred, target[pick])

        try:
            ref_data = next(ref_iterloader)
        except StopIteration:
            ref_iterloader = iter(ref_loader)
            ref_data = next(ref_iterloader)

        ref_loss = get_coreset_loss(model, ref_data, config)
        loss += config['ref_hyp'] * ref_loss
        loss.backward()
        optimizer.step()

        if is_last_step:
            candidates_indices.append(pick)

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)
        update_coreset(loader, task, model, task_id, config)

    return model


def train_coreset_single_step(model, optimizer, loader, task, step, config):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    is_last_step = True if step == config['n_substeps'] else False
    prev_tids=[]
    prev_coreset = [loader['coreset'][tid]['train'].data for tid in range(1, task)]
    prev_targets = [loader['coreset'][tid]['train'].targets for tid in range(1, task)]
    c_x = torch.cat(prev_coreset, 0)
    c_y = torch.cat(prev_targets, 0)

    ref_loader = reconstruct_coreset_loader2(config, loader['coreset'], task-1)
    ref_iterloader = iter(ref_loader)
    rs = np.random.RandomState(0)
    if config['select_type'] in coreset_methods and config['select_type'] != 'coreset':
        summarizer = Summarizer.factory(config['select_type'], rs)

    candidates_indices=[]
    for batch_idx, (data, target, task_id) in enumerate(loader['sequential'][task]['train']):
        model.train()
        optimizer.zero_grad()

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        size = min(config['batch_size'],len(data))
        pick = summarizer.build_summary(data.cpu().numpy(), target.cpu().numpy(), size, method=config['select_type'], model=model, device=DEVICE)
        pred = model(data[pick], task_id)
        loss = criterion(pred, target[pick])
        try:
            ref_data = next(ref_iterloader)
        except StopIteration:
            ref_iterloader = iter(ref_loader)
            ref_data = next(ref_iterloader)
        ref_loss = get_coreset_loss(model, ref_data, config)
        loss += config['ref_hyp'] * ref_loss
        loss.backward()

        if is_last_step:
            if config['select_type'] == 'coreset':
                if batch_idx < 10:
                    selected_pick, _, = bc.build_with_representer_proxy_batch(data.cpu().numpy(), target.cpu().numpy(), config['batch_size'], kernel_fn, cache_kernel=True,
                                                                    start_size=1, inner_reg=1e-3)
            candidates_indices.append(pick)
        optimizer.step()

    if is_last_step:
        select_coreset(loader, task, model, candidates_indices, config)
        update_coreset(loader, task, model, task_id, config)
    return model

def eval_single_epoch(net, loader, config):
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    count = 0 # because of sampler
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for data, target, task_id in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)

            count += len(target)
            output = net(data, task_id)

            test_loss += criterion(output, target).item()*len(target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            correct_bool = pred.eq(target.data.view_as(pred))
            for cid in range(10):
                cid_index = torch.where(target==cid, torch.ones_like(target), torch.zeros_like(target))
                class_correct[cid] += (cid_index.data.view_as(correct_bool) * correct_bool).sum().item()
                class_total[cid] += cid_index.sum().item()
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count

    pc_avg_acc = [np.round(a/(b+1e-10), 4) for a,b in zip(class_correct, class_total)]
    return {'accuracy': avg_acc, 'per_class_accuracy':pc_avg_acc, 'loss': test_loss}

def train_task_sequentially(task, train_loader, config, summary=None):
    EXP_DIR = config['exp_dir']

    current_lr = config['seq_lr'] * (config['lr_decay'])**(task-1)
    prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
    prev_model_path = '{}/{}.pth'.format(EXP_DIR, prev_model_name)
    model = load_model(prev_model_path).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=config['momentum'])

    if task == 1 and config['ocspick']:
        autograd_hacks.add_hooks(model)

    config['n_substeps'] = int(config['seq_epochs'] * (config['stream_size'] / config['batch_size']))
    for _step in range(1, config['n_substeps']+1):
        if config['coreset_base'] and task > 1:
            model = train_coreset_single_step(model, optimizer, train_loader, task, _step, config)
        elif task == 1 or (config['ocspick'] == False):
            model = train_single_step(model, optimizer, train_loader, task, _step, config)
        else:
            model = train_ocs_single_step(model, optimizer, train_loader, task, _step, config)
        metrics = eval_single_epoch(model, train_loader['sequential'][task]['val'], config)
        print('Epoch {} >> (per-task accuracy): {}'.format(_step/config['n_substeps'], np.mean(metrics['accuracy'])))
        print('Epoch {} >> (class accuracy): {}'.format(_step/config['n_substeps'], metrics['per_class_accuracy']))
    return model
