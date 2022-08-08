import torch
import numpy as np

def centering(K):
    n = K.shape[0]
    unit = torch.ones([n, n])
    I = torch.eye(n)
    H = I - unit / n
    return (H*K)* H

def linear_HSIC(X, Y):
    gram_X = torch.matmul(X, (X.T))
    gram_Y = torch.matmul(Y , (Y.T))
    return torch.sum(centering(gram_X) * centering(gram_Y))


def linear_CKA(X, Y):
    # for conv layers, average accross H, W
    if len(X.shape) == 4:
        X = torch.mean(X, dim=(1, 2))
        Y = torch.mean(Y, dim=(1, 2))
    hsic = linear_HSIC(X, Y)
    var1 = torch.sqrt(linear_HSIC(X, X))
    var2 = torch.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def calculate_CKA(m1, m2, eval_loader, num_batches):
    # num_batches = 1
    m1.save_acts, m2.save_acts = True, True
    m1.eval()
    m2.eval()

    layer_keys, sim_scores = None, None
    n_batches = 0
    for batch_idx, (data, target, task_id) in enumerate(eval_loader):
        n_batches += 1
        if batch_idx == num_batches:
            break
        m1(data, task_id)
        m2(data, task_id)
        if sim_scores is None:
            layer_keys = list(m1.acts.keys())
            sim_scores = np.zeros((len(layer_keys), len(layer_keys)))
        for i, k1 in enumerate(layer_keys):
            for j, k2 in enumerate(layer_keys):
                sim_scores[i][j] += linear_CKA(m1.acts[k1], m2.acts[k2]).item()

    # reset model setting
    m1.save_acts, m2.save_acts = False, False
    return sim_scores/(1.0*n_batches), layer_keys

# if __name__ == "__main__":
#     x = torch.randn((32, 100))
#     y = x.clone()
#     print("LCKA >> ", linear_CKA(x, y))

#     m1.save_acts = True
#     m2.save_acts = True

#     m1.eval()
#     m2.eval()

#     inp = torch.randn((16, 32, 32, 3))
#     o1 = m1(inp, 1)
#     o2 = m2(inp, 1)

#     print(linear_CKA(m1.acts['block_1'], m2.acts['block_2']))


