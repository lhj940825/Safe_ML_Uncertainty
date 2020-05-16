import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as transforms
import sklearn.datasets as datasets
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Old method, will be updated or deprecated
def create_dataloader(data_path, batch_size, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader

#Old method, will be updated or deprecated
def create_test_dataloader(data_path, batch_size=1, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return testloader

def data_to_torch_dataset(data, target):
    data = torch.tensor(data, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.float).view(-1, 1)
    # target = torch.tensor(target, dtype=torch.long).view(-1, 1)
    return torch.utils.data.TensorDataset(data, target)

def boston_dataset():
    boston = datasets.load_boston()
    data = boston.data
    target = boston.target
    feature_dim = len(boston.feature_names)
    boston = data_to_torch_dataset(data, target)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0), std=(1.0))])
    return data_to_torch_dataset(data, target), feature_dim

class UCIDataset(Dataset):
    def __init__(self, data_path, transform=None, testing=False):
        data = np.genfromtxt(data_path, delimiter=",") # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        # #Keep the stat params for normalization.
        self.stat = np.asarray([np.mean(self.Y, axis=0), np.std(self.Y, axis=0)])

        # Computation of prior variance and noise from:
        # https://github.com/HIPS/Probabilistic-Backpropagation/blob/60ece68fe535b3b9d74cc71f996145e982872f2e/theano/PBP_net/prior.py
        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * self.stat[1]
        a_sigma_hat_nat = a_sigma - 1
        b_sigma_hat_nat = -b_sigma
        self.prior = np.asarray([a_sigma_hat_nat + 1, -b_sigma_hat_nat])

        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))
        self.X = scaler.fit_transform(self.X)
        # self.Y = scaler.fit_transform(self.Y.reshape(-1, 1))

        # code for normalization new normalization
        # if testing:
        #     self.Y = self.Y * self.stat[1] + self.stat[0]

        self.input_dim = len(self.X[0])

        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        rtn_dict = {}

        x = np.asarray(self.X[idx])
        y = np.asarray(self.Y[idx])

        if self.transform:
            x = self.transform(x)

        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(y, dtype=torch.float).view(-1)

        rtn_dict["input"] = x
        rtn_dict["target"] = y
        rtn_dict["stat"] = self.stat
        rtn_dict["prior"] = self.prior

        # return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1)# return in form of tensor
        # return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1), self.stat # code for normalization new normalization
        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["input", "target"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
        rtn_dict["stat"] = self.stat
        rtn_dict["prior"] = self.prior

        return rtn_dict

class WineDataset(Dataset):
    def __init__(self, data_path, transform=None, testing=False):
        data = np.genfromtxt(data_path, delimiter=",")[1:] # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))
        self.X = scaler.fit_transform(self.X)
        # self.Y = scaler.fit_transform(self.Y.reshape(-1, 1))
        self.input_dim = len(self.X[0])

        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx])
        y = np.asarray(self.Y[idx])

        if self.transform:
            x = self.transform(x.reshape(1, -1))

        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(y, dtype=torch.float).view(-1)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1) #return in form of tensor

def dataset_split(dataset, fname, tar_split=-1, split_ratio=0.1):
    X = dataset[:, :tar_split]
    Y = dataset[:, tar_split]
    # Y = dataset[:, tar_split:] #Currently we only use one target feature for all datasets. This line is for multi targets and is commented

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=2020, shuffle=True)
    train = np.hstack([X_train, Y_train.reshape(len(X_train), -1)])
    test = np.hstack([X_test, Y_test.reshape(len(X_test), -1)])

    np.savetxt(fname + "_train.csv", train, delimiter=",", fmt="%10.6f")
    np.savetxt(fname + "_test.csv", test, delimiter=",", fmt="%10.6f")

    train = np.genfromtxt(fname + "_train.csv", delimiter=',')
    test = np.genfromtxt(fname + "_test.csv", delimiter=',')
    mean_train = np.mean(train, axis=0)
    std_train = np.std(train, axis=0)
    mean_test = np.mean(test, axis=0)
    std_test = np.std(test, axis=0)

if __name__ == "__main__":
    data_dirs = {}
    target_splits = {}
    data_dirs["energy"] = os.path.join("./..", "data", "energy")
    target_splits["energy"] = -2
    data_dirs["kin8nm"] = os.path.join("./..", "data", "kin8nm")
    target_splits["kin8nm"] = -1
    data_dirs["protein"] = os.path.join("./..", "data", "protein")
    target_splits["protein"] = -1
    # data_dirs["naval"] = os.path.join("./..", "data", "naval")
    # data_dirs["yacht"] = os.path.join("./..", "data", "yacht")

    data_files = {}
    datasets = {}
    for key, data_dir in data_dirs.items():
        os.makedirs(data_dir, exist_ok=True)
        data_files[key] = os.path.join(data_dir, key + ".csv")
        datasets[key] = np.genfromtxt(data_files[key], delimiter=',')[1:]
        mean = np.mean(datasets[key], axis=0)
        std = np.std(datasets[key], axis=0)
        dataset_split(datasets[key], os.path.join(data_dir, key), tar_split=target_splits[key])

    naval_dir = os.path.join("./..", "data", "naval")
    yacht_dir = os.path.join("./..", "data", "yacht")
    naval = np.genfromtxt(os.path.join(naval_dir, "naval.txt"))
    yacht = np.genfromtxt(os.path.join(yacht_dir, "yacht.data"))
    dataset_split(naval, os.path.join(naval_dir, "naval"), tar_split=-2)
    dataset_split(yacht, os.path.join(yacht_dir, "yacht"))

    print("finished")

#     def __getitem__(self, item):
#         x = np.asarray(self.X[item])
#         y = np.asarray(self.Y[item])
#
#         if self.transform is not None:
#             x = self.transform(x)
#
#         return torch.from_numpy(x), torch.from_numpy(y) #return in form of tensor
#
#
# if __name__ == '__main__':
#     #split_wine_dataset('./dataset/wine/winequality-white.csv')
#     dataset = WineDataset('../dataset/wine/test_winequality-white.csv', None)
#     data_loader = DataLoader(dataset,batch_size=1, shuffle=True, num_workers=0)
#     print((next(iter(data_loader))))
#     print(np.shape(next(iter(data_loader))[0]))
# >>>>>>> lhj/wineDataset
