import numpy as np
import sklearn
import sklearn.datasets


class ToyDataset(object):
    def __init__(self, dim, data_file=None, static_data=None):
        if data_file is not None:
            self.static_data = np.load(data_file)
        elif static_data is not None:
            self.static_data = static_data
        else:
            self.static_data = None
        self.dim = dim

    def gen_batch(self, batch_size):
        raise NotImplementedError

    def data_gen(self, batch_size, auto_reset):
        if self.static_data is not None:
            num_obs = self.static_data.shape[0]
            while True:
                for pos in range(0, num_obs, batch_size):
                    if pos + batch_size > num_obs:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = num_obs - pos
                    else:
                        num_samples = batch_size
                    yield self.static_data[pos : pos + num_samples, :]
                if not auto_reset:
                    break
                np.random.shuffle(self.static_data)
        else:
            while True:
                yield self.gen_batch(batch_size)


class OnlineToyDataset(ToyDataset):
    def __init__(self, data_name, discrete_dim=16):
        super(OnlineToyDataset, self).__init__(2)
        assert discrete_dim % 2 == 0
        self.data_name = data_name
        self.rng = np.random.RandomState()

        rng = np.random.RandomState(1)
        samples = inf_train_gen(self.data_name, rng, 5000)
        self.f_scale = np.max(np.abs(samples)) + 1   # for normalization
        self.int_scale = 2 ** (discrete_dim / 2 - 1) / (self.f_scale + 1)
        print('f_scale,', self.f_scale, 'int_scale,', self.int_scale)

    def gen_batch(self, batch_size):
        return inf_train_gen(self.data_name, self.rng, batch_size)

    def gen_batch_with_seed(self, batch_size, seed):
        rng = np.random.RandomState(seed)
        return inf_train_gen(self.data_name, rng, batch_size)


# Dataset iterator
def inf_train_gen(data, rng=None, batch_size=200):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0, random_state=rng)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data

    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08, random_state=rng)[0]
        data = data.astype("float32")
        data *= 3
        return data

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1, random_state=rng)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        return data

    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        return 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))

    elif data == "2spirals":
        # n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        # d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        # d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        # x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        # x += np.random.randn(*x.shape) * 0.1

        n = np.sqrt(rng.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + rng.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += rng.randn(*x.shape) * 0.1
        return x

    elif data == "checkerboard":
        # x1 = np.random.rand(batch_size) * 4 - 2
        # x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        # x2 = x2_ + (np.floor(x1) % 2)

        x1 = rng.rand(batch_size) * 4 - 2
        x2_ = rng.rand(batch_size) - rng.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        return np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)
    else:
        raise NotImplementedError