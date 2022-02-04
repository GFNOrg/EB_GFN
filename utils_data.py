import torch
import torch.utils.data as data_utils
import torchvision

import numpy as np
import ipdb
import os


# ======================================================================================================================
def load_dynamic_mnist(args, **kwargs):
    # set args
    if args.down_sample:
        args.input_size = [1, 14, 14]
    else:
        args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    from torchvision import datasets, transforms
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/zhangdh/data', train=True, download=True,
          transform=transforms.Compose([transforms.ToTensor()])), batch_size=args.batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/zhangdh/data', train=False,
         transform=transforms.Compose([transforms.ToTensor()])), batch_size=args.batch_size, shuffle=True)

    # preparing data
    x_train = train_loader.dataset.train_data.float().numpy() / 255.
    x_test = test_loader.dataset.test_data.float().numpy() / 255.
    if args.down_sample:
        x_train = x_train[:, ::2, ::2]
        x_test = x_test[:, ::2, ::2]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    y_train = np.array(train_loader.dataset.train_labels.float().numpy(), dtype=int)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))
    y_test = np.array(test_loader.dataset.test_labels.float().numpy(), dtype=int)

    # validation set
    x_val = x_train[50000:60000]
    y_val = np.array(y_train[50000:60000], dtype=int)
    x_train = x_train[0:50000]
    y_train = np.array(y_train[0:50000], dtype=int)

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args



# ======================================================================================================================
def load_dataset(args, **kwargs):
    assert args.data in ['dynamic_mnist', "dmnist"]
    train_loader, val_loader, test_loader, args = load_dynamic_mnist(args, **kwargs)

    return train_loader, val_loader, test_loader, args
