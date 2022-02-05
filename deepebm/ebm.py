import numpy as np
import torch
import torch.nn as nn
import torchvision
import os, sys
import copy
import time
import random
import ipdb
from tqdm import tqdm
import argparse
import network

sys.path.append("/home/zhangdh/EB_GFN")
from gflownet import get_GFlowNet
import utils_data


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")

class EBM(nn.Module):
    def __init__(self, net, mean=None):
        super().__init__()
        self.net = net
        if mean is None:
            self.mean = None
        else:
            self.mean = nn.Parameter(mean, requires_grad=False)
            self.base_dist = torch.distributions.Bernoulli(probs=self.mean)

    def forward(self, x):
        if self.mean is None:
            bd = 0.
        else:
            bd = self.base_dist.log_prob(x).sum(-1)

        logp = self.net(x).squeeze()
        return logp + bd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=0, type=int)
    # data
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--data', type=str, default='dmnist')
    parser.add_argument("--down_sample", "--ds", default=0, type=int, choices=[0, 1])
    parser.add_argument('--ckpt_path', type=str, default=None)
    # models
    parser.add_argument('--model', type=str, default='mlp-256')
    parser.add_argument('--base_dist', "--bd", type=int, default=1, choices=[0, 1])
    parser.add_argument('--gradnorm', "--gn", type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--n_iters', "--ni", type=lambda x: int(float(x)), default=5e4)
    parser.add_argument('--batch_size', "--bs", type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--print_every', "--pe", type=int, default=100)
    parser.add_argument('--viz_every', "--ve", type=int, default=2000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument("--ebm_every", "--ee", type=int, default=1, help="EBM training frequency")

    # for GFN
    parser.add_argument("--type", type=str)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--hid_layers", "--hl", type=int, default=5)
    parser.add_argument("--leaky", type=int, default=1, choices=[0, 1])
    parser.add_argument("--gfn_bn", "--gbn", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init_zero", "--iz", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gmodel", "--gm", type=str, default="mlp")
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
    parser.add_argument("--zlr", type=float, default=1e-1)
    parser.add_argument("--momentum", "--mom", type=float, default=0.0)
    parser.add_argument("--gfn_weight_decay", "--gwd", type=float, default=0.0)
    parser.add_argument('--mc_num', "--mcn", type=int, default=5)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

    args.device = device
    args.save_dir = os.path.join(args.save_dir, "test")
    makedirs(args.save_dir)

    print("Device:" + str(device))
    print("Args:" + str(args))

    before_load = time.time()
    train_loader, val_loader, test_loader, args = utils_data.load_dataset(args)
    plot = lambda p, x: torchvision.utils.save_image(x.view(x.size(0), args.input_size[0],
        args.input_size[1], args.input_size[2]), p, normalize=True, nrow=int(x.size(0) ** .5))
    print(f"It takes {time.time() - before_load:.3f}s to load {args.data} dataset.")

    def preprocess(data):
        if args.dynamic_binarization:
            return torch.bernoulli(data)
        else:
            return data

    if args.down_sample:
        assert args.model.startswith("mlp-")

    if args.model.startswith("mlp-"):
        nint = int(args.model.split('-')[1])
        net = network.mlp_ebm(np.prod(args.input_size), nint)
    elif args.model.startswith("cnn-"):
        nint = int(args.model.split('-')[1])
        net = network.MNISTConvNet(nint)
    elif args.model.startswith("resnet-"):
        nint = int(args.model.split('-')[1])
        net = network.ResNetEBM(nint)
    else:
        raise ValueError("invalid model definition")

    init_batch = []
    for x, _ in train_loader:
        init_batch.append(preprocess(x))
    init_batch = torch.cat(init_batch, 0)
    eps = 1e-2
    init_mean = init_batch.mean(0) * (1. - 2 * eps) + eps

    if args.base_dist:
        model = EBM(net, init_mean)
    else:
        model = EBM(net)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    xdim = np.prod(args.input_size)
    assert args.gmodel == "mlp"
    gfn = get_GFlowNet(args.type, xdim, args, device)
    model.to(device)
    print("model: {:}".format(model))

    itr = 0
    while itr < args.n_iters:
        for x in train_loader:
            st = time.time()
            x = preprocess(x[0].to(device))  #  -> (bs, 784)

            if args.gradnorm > 0:
                x.requires_grad_()

            update_success_rate = -1.
            assert "tb" in args.type
            train_loss, train_logZ = gfn.train(args.batch_size, scorer=lambda inp: model(inp).detach(),
                   silent=itr % args.print_every != 0, data=x, back_ratio=args.back_ratio)

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
                    lp_update = model(x_fake).squeeze() - model(x).squeeze()
                    update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                    updates = update_dist.sample()
                    x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                    update_success_rate = updates.mean().item()

            else:
                x_fake = gfn.sample(args.batch_size)

            if itr % args.ebm_every == 0:
                st = time.time() - st

                model.train()
                logp_real = model(x).squeeze()
                if args.gradnorm > 0:
                    grad_ld = torch.autograd.grad(logp_real.sum(), x,
                                      create_graph=True)[0].flatten(start_dim=1).norm(2, 1)
                    grad_reg = (grad_ld ** 2. / 2.).mean()
                else:
                    grad_reg = torch.tensor(0.).to(device)

                logp_fake = model(x_fake).squeeze()
                obj = logp_real.mean() - logp_fake.mean()
                l2_reg = (logp_real ** 2.).mean() + (logp_fake ** 2.).mean()
                loss = -obj + grad_reg * args.gradnorm + args.l2 * l2_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if itr % args.print_every == 0 or itr == args.n_iters - 1:
                print("({:5d}) | ({:.3f}s/iter) |log p(real)={:.2e}, "
                     "log p(fake)={:.2e}, diff={:.2e}, grad_reg={:.2e}, l2_reg={:.2e} update_rate={:.1f}".format(itr, st,
                     logp_real.mean().item(), logp_fake.mean().item(), obj.item(), grad_reg.item(), l2_reg.item(), update_success_rate))

            if (itr + 1) % args.eval_every == 0:
                model.eval()
                print("GFN TEST")
                gfn.model.eval()
                gfn_test_ll = gfn.evaluate(test_loader, preprocess, args.mc_num)
                print("GFN Test log-likelihood ({}) with {} samples: {}".format(itr, args.mc_num, gfn_test_ll.item()))

                model.cpu()
                d = {}
                d['model'] = model.state_dict()
                d['optimizer'] = optimizer.state_dict()
                gfn_ckpt = {"model": gfn.model.state_dict(), "optimizer": gfn.optimizer.state_dict(),}
                gfn_ckpt["logZ"] = gfn.logZ.detach().cpu()
                torch.save(d, "{}/ckpt.pt".format(args.save_dir))
                torch.save(gfn_ckpt, "{}/gfn_ckpt.pt".format(args.save_dir))

                model.to(device)

            itr += 1
            if itr > args.n_iters:
                print("Training finished!")
                quit(0)
