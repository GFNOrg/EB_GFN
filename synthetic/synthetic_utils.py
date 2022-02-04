import torch as T
import torch
import torch.nn.functional as F

import numpy as np
import tqdm
import random
import sys, os
from matplotlib import pyplot as plt
from sympy.combinatorics.graycode import GrayCode
import time
import ipdb



def get_true_samples(db, size, bm, int_salce, discrete_dim, seed=None):
    if seed is None:
        samples = float2bin(db.gen_batch(size), bm, int_salce, discrete_dim)
    else:
        samples = float2bin(db.gen_batch_with_seed(size, seed), bm, int_salce, discrete_dim)
    return torch.from_numpy(samples).float()

def get_ebm_samples(score_func, size, inv_bm, int_scale, discrete_dim, device, gibbs_sampler=None, gibbs_steps=20):
    unif_dist = torch.distributions.Bernoulli(probs=0.5)
    ebm_samples = unif_dist.sample((size, discrete_dim)).to(device)
    ebm_samp_float = []
    for ind in range(gibbs_steps * discrete_dim):  # takes about 1s
        ebm_samples = gibbs_sampler.step(ebm_samples, score_func)
    ebm_samp_float.append(bin2float(ebm_samples.data.cpu().numpy().astype(int), inv_bm, int_scale, discrete_dim))
    ebm_samp_float = np.concatenate(ebm_samp_float, axis=0)
    return ebm_samples, ebm_samp_float

def estimate_ll(score_func, samples, n_partition=None, rand_samples=None):
    with torch.no_grad():
        if rand_samples is None:
            rand_samples = torch.randint(2, (n_partition, samples.shape[1])).float().to(samples.device)
        n_partition = rand_samples.shape[0]
        f_z_list = []
        for i in range(0, n_partition, samples.shape[0]):  # 从0数到n_partition，每一份是samples.shape[0]大小
            f_z = score_func(rand_samples[i:i+samples.shape[0]]).view(-1, 1)
            f_z_list.append(f_z)
        f_z = torch.cat(f_z_list, dim=0)
        f_z = f_z - samples.shape[1] * np.log(0.5) - np.log(n_partition)  # log(1/2)是unif的概率，importance sample的时候在分母

        # log_part = logsumexp(f_z)
        log_part = f_z.logsumexp(0)
        f_sample = score_func(samples)
        ll = f_sample - log_part

    return torch.mean(ll).item()


def exp_hamming_sim(x, y, bd):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = T.sum(T.abs(x - y), dim=-1)
    return T.exp(-bd * d)


def exp_hamming_mmd(x, y, bandwidth=0.1):
    x = x.float()
    y = y.float()

    with T.no_grad():
        kxx = exp_hamming_sim(x, x, bd=bandwidth)
        idx = T.arange(0, x.shape[0], out=T.LongTensor())
        kxx[idx, idx] = 0.0
        kxx = T.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = exp_hamming_sim(y, y, bd=bandwidth)
        idx = T.arange(0, y.shape[0], out=T.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = T.sum(kyy) / y.shape[0] / (y.shape[0] - 1)

        kxy = T.sum(exp_hamming_sim(x, y, bd=bandwidth)) / x.shape[0] / y.shape[0]

        mmd = kxx + kyy - 2 * kxy
    return mmd


def hamming_sim(x, y):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    d = torch.sum(torch.abs(x - y), dim=-1)
    return x.shape[-1] - d

def hamming_mmd(x, y):
    x = x.float()
    y = y.float()
    with torch.no_grad():
        kxx = hamming_sim(x, x)
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx[idx, idx] = 0.0
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = hamming_sim(y, y)
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(hamming_sim(x, y)) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd


def linear_mmd(x, y):
    x = x.float()
    y = y.float()
    with torch.no_grad():
        kxx = torch.mm(x, x.transpose(0, 1))
        idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
        kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
        kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

        kyy = torch.mm(y, y.transpose(0, 1))
        idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
        kyy[idx, idx] = 0.0
        kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
        kxy = torch.sum(torch.mm(y, x.transpose(0, 1))) / x.shape[0] / y.shape[0]
        mmd = kxx + kyy - 2 * kxy
    return mmd


from torch.autograd import Variable, Function
def get_gamma(X, bandwidth):
    with torch.no_grad():
        x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        x_t = torch.transpose(X, 0, 1)
        x_norm_t = x_norm.view(1, -1)
        t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
        dist2 = F.relu(Variable(t)).detach().data

        d = dist2.cpu().numpy()
        d = d[np.isfinite(d)]
        d = d[d > 0]
        median_dist2 = float(np.median(d))
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def get_kernel_mat(x, landmarks, gamma):
    d = pairwise_distances(x, landmarks)
    k = torch.exp(d * -gamma)
    k = k.view(x.shape[0], -1)
    return k

def MMD(x, y, bandwidth=1.0):
    y = y.detach()
    gamma = get_gamma(x.detach(), bandwidth)
    kxx = get_kernel_mat(x, x, gamma)
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    kxx = kxx * (1 - torch.eye(x.shape[0]).to(x.device))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    kyy[idx, idx] = 0.0
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd



def get_binmap(discrete_dim, binmode):
    b = discrete_dim // 2 - 1
    all_bins = []
    for i in range(1 << b):
        bx = np.binary_repr(i, width=discrete_dim // 2 - 1)
        all_bins.append('0' + bx)
        all_bins.append('1' + bx)
    vals = all_bins[:]
    if binmode == 'rand':
        print('remapping binary repr with random permute')
        random.shuffle(vals)
    elif binmode == 'gray':
        print('remapping binary repr with gray code')
        a = GrayCode(b)
        vals = []
        for x in a.generate_gray():
            vals.append('0' + x)
            vals.append('1' + x)
    else:
        assert binmode == 'normal'
    bm = {}
    inv_bm = {}
    for i, key in enumerate(all_bins):
        bm[key] = vals[i]
        inv_bm[vals[i]] = key
    return bm, inv_bm


def compress(x, discrete_dim):
    bx = np.binary_repr(int(abs(x)), width=discrete_dim // 2 - 1)
    bx = '0' + bx if x >= 0 else '1' + bx
    return bx


def recover(bx):
    x = int(bx[1:], 2)
    return x if bx[0] == '0' else -x


def float2bin(samples, bm, int_scale, discrete_dim):
    bin_list = []
    for i in range(samples.shape[0]):
        x, y = samples[i] * int_scale
        bx, by = compress(x, discrete_dim), compress(y, discrete_dim)
        bx, by = bm[bx], bm[by]
        bin_list.append(np.array(list(bx + by), dtype=int))
    return np.array(bin_list)


def bin2float(samples, inv_bm, int_scale, discrete_dim):
    floats = []
    for i in range(samples.shape[0]):
        s = ''
        for j in range(samples.shape[1]):
            s += str(samples[i, j])
        x, y = s[:discrete_dim // 2], s[discrete_dim // 2:]
        x, y = inv_bm[x], inv_bm[y]
        x, y = recover(x), recover(y)
        x /= int_scale
        y /= int_scale
        floats.append((x, y))
    return np.array(floats)


def plot_heat(score_func, bm, size, device, int_scale, discrete_dim, out_file=None):
    w = 100
    x = np.linspace(-size, size, w)
    y = np.linspace(-size, size, w)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, [-1, 1])
    yy = np.reshape(yy, [-1, 1])
    heat_samples = float2bin(np.concatenate((xx, yy), axis=-1), bm, int_scale, discrete_dim)
    heat_samples = torch.from_numpy(heat_samples).to(device).float()
    heat_score = F.softmax(score_func(heat_samples).view(1, -1), dim=-1)
    a = heat_score.view(w, w).data.cpu().numpy()
    a = np.flip(a, axis=0)
    print("energy max and min:", a.max(), a.min())
    plt.imshow(a)
    plt.axis('equal')
    plt.axis('off')
    # if out_file is None:
    #     out_file = os.path.join(save_dir, 'heat.pdf')
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def plot_samples(samples, out_name, lim=None, axis=True):
    plt.scatter(samples[:, 0], samples[:, 1], marker='.')
    plt.axis('equal')
    if lim is not None:
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
    if not axis:
        plt.axis('off')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()


############# Model Architecture

class EnergyModel(T.nn.Module):

    def __init__(self, s, mid_size):
        super(EnergyModel, self).__init__()

        self.m = T.nn.Sequential(T.nn.Linear(s, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, mid_size),
                                 T.nn.ELU(),
                                 T.nn.Linear(mid_size, 1))

    def forward(self, x):
        x = x.view((x.shape[0], -1))
        x = self.m(x)

        return x[:, -1]
