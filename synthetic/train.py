import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, sys

import time
import random
import ipdb, pdb
from tqdm import tqdm
import argparse

sys.path.append("/home/zhangdh/EB_GFN")
from gflownet import get_GFlowNet

sys.path.append("/home/zhangdh/EB_GFN/synthetic")
from synthetic_utils import plot_heat, plot_samples,\
    float2bin, bin2float, get_binmap, get_true_samples, get_ebm_samples, EnergyModel
from synthetic_data import inf_train_gen, OnlineToyDataset


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


unif_dist = torch.distributions.Bernoulli(probs=0.5)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    # data
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--data', type=str, default='circles')  # 2spirals 8gaussians pinwheel circles moons swissroll checkerboard

    # training
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=1e5)
    parser.add_argument('--batch_size', "--bs", type=int, default=128)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument("--ebm_every", "--ee", type=int, default=1, help="EBM training frequency")

    # for GFN
    parser.add_argument("--type", type=str)
    parser.add_argument("--hid", type=int, default=512)
    parser.add_argument("--hid_layers", "--hl", type=int, default=3)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1], )
    parser.add_argument("--gmodel", "--gm", type=str,default="mlp")
    parser.add_argument("--train_steps", "--ts", type=int, default=1)
    parser.add_argument("--l1loss", "--l1l", type=int, default=0, choices=[0, 1], help="use soft l1 loss instead of l2")

    parser.add_argument("--with_mh", "--wm", type=int, default=0, choices=[0, 1])
    parser.add_argument("--rand_k", "--rk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--lin_k", "--lk", type=int, default=0, choices=[0, 1])
    parser.add_argument("--warmup_k", "--wk", type=lambda x: int(float(x)), default=0, help="need to use w/ lin_k")
    parser.add_argument("--K", type=int, default=-1, help="for gfn back forth negative sample generation")

    parser.add_argument("--rand_coef", "--rc", type=float, default=0, help="for tb")
    parser.add_argument("--back_ratio", "--br", type=float, default=0.)
    parser.add_argument("--clip", type=float, default=-1., help="for gfn's linf gradient clipping")
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--glr", type=float, default=1e-3)
    parser.add_argument("--zlr", type=float, default=1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

    args.save_dir = os.path.join(args.save_dir, "test")
    makedirs(args.save_dir)

    print("Device:" + str(device))
    print("Args:" + str(args))

    ############## Data
    discrete_dim = 32
    bm, inv_bm = get_binmap(discrete_dim, 'gray')

    db = OnlineToyDataset(args.data, discrete_dim)
    if not hasattr(args, "int_scale"):
        int_scale = db.int_scale
    else:
        int_scale = args.int_scale
    if not hasattr(args, "plot_size"):
        plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
        plot_size = args.plot_size
    # plot_size = 4.1

    batch_size = args.batch_size
    multiples = {'pinwheel': 5, '2spirals': 2}
    batch_size = batch_size - batch_size % multiples.get(args.data, 1)

    ############## EBM model
    energy_model = EnergyModel(discrete_dim, 256).to(device)
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=args.lr)

    ############## GFN
    xdim = discrete_dim
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, device)

    energy_model.to(device)
    print("model: {:}".format(energy_model))

    itr = 0
    best_val_ll = -np.inf
    best_itr = -1
    lr = args.lr
    while itr < args.n_iters:
        st = time.time()

        x = get_true_samples(db, batch_size, bm, int_scale, discrete_dim).to(device)

        update_success_rate = -1.
        gfn.model.train()
        train_loss, train_logZ = gfn.train(batch_size,
                scorer=lambda inp: energy_model(inp).detach(), silent=itr % args.print_every != 0, data=x,
                back_ratio=args.back_ratio)

        if args.rand_k or args.lin_k or (args.K > 0):
            if args.rand_k:
                K = random.randrange(xdim) + 1
            elif args.lin_k:
                K = min(xdim, int(xdim * float(itr + 1) / args.warmup_k))
                K = max(K, 1)
            elif args.K > 0:
                K = args.K
            else:
                raise ValueError

            gfn.model.eval()
            x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

            delta_logp_traj = delta_logp_traj.detach()
            if args.with_mh:
                # MH step, calculate log p(x') - log p(x)
                lp_update = energy_model(x_fake).squeeze() - energy_model(x).squeeze()
                update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                updates = update_dist.sample()
                x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                update_success_rate = updates.mean().item()

        else:
            x_fake = gfn.sample(batch_size)


        if itr % args.ebm_every == 0:
            st = time.time() - st

            energy_model.train()
            logp_real = energy_model(x).squeeze()

            logp_fake = energy_model(x_fake).squeeze()
            obj = logp_real.mean() - logp_fake.mean()
            l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
            loss = -obj

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if itr % args.print_every == 0 or itr == args.n_iters - 1:
            print("({:5d}) | ({:.3f}s/iter) cur lr= {:.2e} |log p(real)={:.2e}, "
                     "log p(fake)={:.2e}, diff={:.2e}, update_rate={:.1f}".format(
                itr, st, lr, logp_real.mean().item(), logp_fake.mean().item(), obj.item(), update_success_rate))


        if (itr + 1) % args.eval_every == 0:
            # heat map of energy
            plot_heat(energy_model, bm, plot_size, device, int_scale, discrete_dim,
                      out_file=os.path.join(args.save_dir, f'heat_{itr}.pdf'))

            # samples of gfn
            gfn_samples = gfn.sample(4000).detach()
            gfn_samp_float = bin2float(gfn_samples.data.cpu().numpy().astype(int), inv_bm, int_scale, discrete_dim)
            plot_samples(gfn_samp_float, os.path.join(args.save_dir, f'gfn_samples_{itr}.pdf'), lim=plot_size)

            # GFN LL
            gfn.model.eval()
            logps = []
            pbar = tqdm(range(10))
            pbar.set_description("GFN Calculating likelihood")
            for _ in pbar:
                pos_samples_bs = get_true_samples(db, 1000, bm, int_scale, discrete_dim).to(device)
                logp = gfn.cal_logp(pos_samples_bs, 20)
                logps.append(logp.reshape(-1))
                pbar.set_postfix({"logp": f"{torch.cat(logps).mean().item():.2f}"})
            gfn_test_ll = torch.cat(logps).mean()

            print(f"Test NLL ({itr}): GFN: {-gfn_test_ll.item():.3f}")


        itr += 1
        if itr > args.n_iters:
            quit(0)
