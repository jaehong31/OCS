import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from core.utils import compute_and_flatten_example_grads, flatten_example_grads
from core.mode_connectivity import get_ti
import os
from .utils import load_model
import time
import core.autograd_hacks as autograd_hacks


class Summarizer(ABC):

    def __init__(self, rs=None):
        super().__init__()
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs

    @abstractmethod
    def build_summary(self, X, y, size, **kwargs):
        pass

    def factory(type, rs):
        if type == 'uniform': return UniformSummarizer(rs)
        if type == 'kmeans_features': return KmeansFeatureSpace(rs)
        if type == 'kmeans_embedding': return KmeansEmbeddingSpace(rs)
        if type == 'kmeans_grads': return KmeansGradSpace(rs)
        if type == 'kcenter_features': return KcenterFeatureSpace(rs)
        if type == 'kcenter_embedding': return KcenterEmbeddingSpace(rs)
        if type == 'kcenter_grads': return KcenterGradSpace(rs)
        if type == 'entropy': return LargestEntropy(rs)
        if type == 'hardest': return HardestSamples(rs)
        if type == 'frcl': return FRCLSelection(rs)
        if type == 'icarl': return ICaRLSelection(rs)
        if type == 'grad_matching': return GradMatching(rs)
        #if type == 'setencoding': return SetEncoding(rs)
        if type == 'gss_greedy': return GradientSampleSelection(rs)
        if type == 'er_mir': return ErMir(rs)
        raise TypeError('Unkown summarizer type ' + type)

    factory = staticmethod(factory)


class UniformSummarizer(Summarizer):

    def build_summary(self, X, y, size, **kwargs):
        n = X.shape[0]
        inds = self.rs.choice(n, size, replace=False)
        return inds


class KmeansFeatureSpace(Summarizer):

    def kmeans_pp(self, X, k, rs):
        n = X.shape[0]
        inds = np.zeros(k).astype(int)
        inds[0] = rs.choice(n)
        dists = np.sum((X - X[inds[0]]) ** 2, axis=1)
        for i in range(1, k):
            ind = rs.choice(n, p=dists / np.sum(dists))
            inds[i] = ind
            dists = np.minimum(dists, np.sum((X - X[ind]) ** 2, axis=1))
        return inds

    def build_summary(self, X, y, size, **kwargs):
        X_flattened = X.reshape((X.shape[0], -1))
        inds = self.kmeans_pp(X_flattened, size, self.rs)
        return inds


class KmeansEmbeddingSpace(KmeansFeatureSpace):

    def get_embedding(self, X, model, device):
        embeddings = []
        with torch.no_grad():
            model.eval()
            for i in range(X.shape[0]):
                data = torch.from_numpy(X[i:i + 1]).float().to(device)
                embedding = model.embed(data)
                embeddings.append(embedding.cpu().numpy())
        return np.vstack(embeddings)

    def build_summary(self, X, y, size, **kwargs):
        embeddings = self.get_embedding(X, kwargs['model'], kwargs['device'])
        inds = self.kmeans_pp(embeddings, size, self.rs)
        return inds


class KmeansGradSpace(KmeansFeatureSpace):

    def get_grads(self, X, y, model, device, taskid=None):
        y_t = torch.from_numpy(y).to(device).long()
        grads = []
        for i in range(X.shape[0]):
            data = torch.from_numpy(X[i:i + 1]).float().to(device)
            output = model(data, taskid)
            loss = F.cross_entropy(output, y_t[i].view(-1))
            gr = torch.autograd.grad(loss, list(model.parameters())[-2:])
            res = np.hstack([g.view(-1).detach().cpu().numpy() for g in gr])
            grads.append(res)
        return np.vstack(grads)

    def build_summary(self, X, y, size, **kwargs):
        grads = self.get_grads(X, y, kwargs['model'], kwargs['device'])
        inds = self.kmeans_pp(grads, size, self.rs)
        return inds


class KcenterFeatureSpace(Summarizer):

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists

    def kcenter(self, X, size):
        dists = np.full(X.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, X, current_id)
        idx = [current_id]

        for i in range(1, size):
            current_id = np.argmax(dists)
            dists = self.update_distance(dists, X, current_id)
            idx.append(current_id)

        return np.hstack(idx)

    def build_summary(self, X, y, size, **kwargs):
        X_flattened = X.reshape((X.shape[0], -1))
        inds = self.kcenter(X_flattened, size)
        return inds


class KcenterEmbeddingSpace(KcenterFeatureSpace, KmeansEmbeddingSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        embeddings = self.get_embedding(X, model, device)
        inds = self.kcenter(embeddings, size)
        return inds


class KcenterGradSpace(KcenterFeatureSpace, KmeansGradSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        grads = self.get_grads(X, y, model, device)
        inds = self.kcenter(grads, size)
        return inds


class LargestEntropy(Summarizer):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        data = torch.from_numpy(X).to(device).float()
        output = F.softmax(model(data).detach().cpu(), dim=1).numpy()
        entropy = -np.sum(output * np.log(output + 1e-20), axis=1)
        inds = entropy.argsort()[-size:][::-1]
        return inds


class HardestSamples(Summarizer):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        data = torch.from_numpy(X).to(device).float()
        output = model(data)
        loss = F.cross_entropy(output, torch.from_numpy(y).long().to(device), reduction='none')
        inds = loss.detach().cpu().numpy().argsort()[-size:][::-1]
        return inds


class FRCLSelection(KmeansEmbeddingSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        embeddings = self.get_embedding(X, model, device)
        K = np.dot(embeddings, embeddings.T)

        def calc_score(ind):
            K_s = K[ind][:, ind].astype(np.float64)
            K_s_inv = np.linalg.inv(K_s)
            aux = 0
            for i in range(len(ind)):
                aux += K_s[i, i] - K_s[i].dot(K_s_inv).dot(K_s[:, i])
            return aux

        inds = np.random.choice(len(y), size, replace=False)
        nr_outer_it = 20
        nr_inner_it = 20

        score = calc_score(inds)
        for outer_it in range(nr_outer_it):
            for i in range(size):
                aux = inds[i]
                for inner_it in range(nr_inner_it):
                    crt = np.random.choice(len(y))
                    while crt in inds:
                        crt = np.random.choice(len(y))
                    inds[i] = crt
                    new_score = calc_score(inds)
                    if new_score < score:
                        score = new_score
                        aux = crt
                inds[i] = aux
        return inds


class ICaRLSelection(KmeansEmbeddingSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        embeddings = self.get_embedding(X, model, device)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        inds = []
        for c in np.unique(y):
            inds_c = []
            selected_inds = np.where(y == c)[0]
            target = np.mean(embeddings[selected_inds], axis=0)
            current_embedding = np.zeros(embeddings.shape[1])
            for i in range(size // len(np.unique(y)) + 1):
                best_score = np.inf
                for candidate in selected_inds:
                    if candidate not in inds_c:
                        score = np.linalg.norm(
                            target - (embeddings[candidate] + current_embedding) / (i + 1))
                        if score < best_score:
                            best_score = score
                            best_ind = candidate
                inds_c.append(best_ind)
                current_embedding = current_embedding + embeddings[best_ind]
            inds.append(inds_c)
        final_inds = []
        cnt = 0
        crt_pos = 0
        while cnt < size:
            for i in range(len(inds)):
                final_inds.append(inds[i][crt_pos])
                cnt += 1
                if cnt == size:
                    break
            crt_pos += 1
        inds = np.array(final_inds)
        return inds


class GradMatching(KmeansGradSpace):

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        taskid = kwargs['taskid']
        embeddings = self.get_grads(X, y, model, device, taskid)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1)[:, np.newaxis] + 1e-8)
        inds = []
        target = np.mean(embeddings, axis=0)
        current_embedding = np.zeros(embeddings.shape[1])
        for i in range(size):
            best_score = np.inf
            for candidate in np.arange(X.shape[0]):
                if candidate not in inds:
                    score = np.linalg.norm(
                        target - (embeddings[candidate] + current_embedding) / (i + 1))
                    if score < best_score:
                        best_score = score
                        best_ind = candidate
            inds.append(best_ind)
            current_embedding = current_embedding + embeddings[best_ind]
        inds = np.array(inds)

        return inds



class GradientSampleSelection(KmeansGradSpace):

    def build_summary(self, X, y, size, task_id=None, **kwargs):
        model = kwargs['model']
        device = kwargs['device']
        #grads = torch.Tensor(self.get_grads(X, y, model, device, full_grads=True)).to(device)
        #grads = compute_and_flatten_example_grads(model, None, torch.Tensor(X).to(device), torch.Tensor(y).long().to(device), task_id)
        time1=time.time()
        #grads = compute_and_flatten_example_grads(model, None, X, y, task_id)

        _eg = []
        criterion2 = nn.CrossEntropyLoss().to(device)

        autograd_hacks.clear_backprops(model)
        pred = model(X, task_id)
        loss = criterion2(pred, y)
        loss.backward()
        autograd_hacks.compute_grad1(model)
        grads = flatten_example_grads(model)
        time2=time.time()
        reference = kwargs['reference']
        C = kwargs['m_score']
        n = 10
        for idx in range(len(grads)):
            g = grads[idx]
            rand_pick = torch.randperm(len(C))[:n]
            autograd_hacks.clear_backprops(model)
            pred = model(reference.data[rand_pick].to(device), task_id)
            loss = criterion2(pred, reference.targets[rand_pick].long().to(device))
            loss.backward()
            autograd_hacks.compute_grad1(model)
            G = flatten_example_grads(model)
            #G = compute_and_flatten_example_grads(model, None, reference.data[rand_pick].to(device), reference.targets[rand_pick].long().to(device), task_id)

            #G = torch.Tensor(G).to(device)
            c = torch.max(torch.matmul(g, G.t())/(torch.norm(g) * torch.norm(G, dim=1))) + 1
            # update coresets
            if c < 1:
                pi = C[rand_pick] / torch.sum(C[rand_pick])
                i = np.random.choice(np.arange(0, n), size=1, p=pi.numpy())[0]
                r = np.random.random(1)
                if r[0] < C[rand_pick[i]]/(C[rand_pick[i]] + c):
                    reference.data[rand_pick[i]] = copy.deepcopy(X[idx].detach())#.to(device)
                    reference.targets[rand_pick[i]] = copy.deepcopy(y[idx].detach())#.to(device)
                    C[rand_pick[i]] = copy.deepcopy(c.cpu().detach())
            else:
                pass
        time4=time.time()
        #print('t1: %s, t3: %s'%(time2-time1, time4-time3))
        return reference, C


class ErMir(Summarizer):
    def overwrite_grad(self, pp, new_grad, grad_dims):
        cnt = 0
        for param in pp():
            param.grad=torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                    param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def get_future_step_parameters(self, this_net, grad_vector, grad_dims, lr, task, exp_dir):
        # new_net = copy.deepcopy(this_net)
        prev_model_name = 'init' if task == 1 else 't_{}_seq'.format(str(task-1))
        prev_model_path = '{}/{}.pth'.format(exp_dir, prev_model_name)
        new_net = load_model(prev_model_path).cuda()
        new_net.load_state_dict(this_net.state_dict())
        self.overwrite_grad(new_net.parameters,grad_vector,grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data=param.data - lr*param.grad.data
        return new_net

    def get_grad_vector(self, pp, grad_dims):
        grads = torch.Tensor(sum(grad_dims)).cuda()
        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def build_summary(self, X, y, size, **kwargs):
        model = kwargs['model']
        config = kwargs['config']
        device = kwargs['device']
        lr = kwargs['lr']
        task = kwargs['task']
        EXP_DIR = config['exp_dir']
        #data = torch.from_numpy(X).to(device).float()
        data = torch.Tensor(X).to(device)
        labels = torch.Tensor(y).long().to(device)

        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())

        # Class-incremental
        if 'mnist' in config['dataset']:
            grad_vector = self.get_grad_vector(model.parameters, grad_dims)
            model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims, lr=lr, task=task, exp_dir=EXP_DIR)
            with torch.no_grad():
                logits_track_pre = model(data, task)
                buffer_hid = model_temp.embed(data)
                logits_track_post = model_temp.W3(buffer_hid)
            pre_loss = F.cross_entropy(logits_track_pre, labels, reduction="none")
            post_loss = F.cross_entropy(logits_track_post, labels, reduction="none")
        # Task-incremental
        else:
            grad_vector = self.get_grad_vector(model.parameters, grad_dims)
            model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims, lr=lr, task=task, exp_dir=EXP_DIR)
            with torch.no_grad():
                logits_track_pre = model(data, None, return_full=True)
                logits_track_post = model_temp(data, None, return_full=True)
            pre_loss = F.cross_entropy(get_ti(logits_track_pre, labels=labels), labels, reduction="none")
            post_loss = F.cross_entropy(get_ti(logits_track_post, labels=labels), labels, reduction="none")

        scores = post_loss - pre_loss
        inds = scores.sort(descending=True)[1][:size]
        return inds
