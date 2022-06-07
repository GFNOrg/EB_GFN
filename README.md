# Energy-based GFlowNets

Code for our ICML 2022 paper [Generative Flow Networks for Discrete Probabilistic Modeling](https://arxiv.org/abs/2202.01361) 
by [Dinghuai Zhang](https://zdhnarsil.github.io/), [Nikolay Malkin](https://malkin1729.github.io/), [Zhen Liu](http://itszhen.com/), 
[Alexandra Volokhova](https://alexandravolokhova.github.io/), Aaron Courville,
[Yoshua Bengio](https://yoshuabengio.org/).


### Example

Synthetic tasks

```
python -m synthetic.train --data checkerboard --lr 1e-3 --type tblb --hid_layer 3 --hid 256 --print_every 100 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 1e5 --with_mh 1
```

Discrete image modeling 

```angular2html
python -m deepebm.ebm --model mlp-256 --lr 1e-4 --type tblb --hid_layer 3 --hid 256 --glr 1e-3 --zlr 1 --rand_coef 0 --back_ratio 0.5 --lin_k 1 --warmup_k 5e4 --with_mh 1 --print_every 100 --mc_num 5
```

