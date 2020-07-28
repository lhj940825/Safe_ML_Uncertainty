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
from sys import platform
if __name__ == '__main__':
    from utils import *
else:
    from .utils import *

def data_to_torch_dataset(data, target):
    data = torch.tensor(data, dtype=torch.float)
    target = torch.tensor(target, dtype=torch.float).view(-1, 1)
    # target = torch.tensor(target, dtype=torch.long).view(-1, 1)
    return torch.utils.data.TensorDataset(data, target)

class UCIDataset(Dataset):
    def __init__(self, data_path, transform=None, testing=False, data_type='normal'):
        data = np.genfromtxt(data_path, delimiter=",") # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        self.Y = data[:, -1]
        # #Keep the stat params for normalization.

        if platform == 'win32':
            dataset_name, data_file = data_path.split('\\')[-2:] #data_path.split('\\')[1] = dataset name, data_path.split('\\')[2] = datafile to load, either train or eval, or test set
        else:
            dataset_name, data_file = data_path.split('/')[-2:] #data_path.split('\\')[1] = dataset name, data_path.split('\\')[2] = datafile to load, either train or eval, or test set

        if not testing:
            mean = np.mean(self.Y, axis=0)
            std = np.std(self.Y, axis=0)
            print("Writing {} train stat to config file: mean_std_{}_dataset.yml".format(dataset_name, data_type))
            store_train_mean_and_std(dataset_name, mean, std, data_type=data_type)
            self.stat = np.asarray([mean, std])
        else:
            yml_dir = os.path.join(os.getcwd(), 'configs/mean_std_{}_dataset.yml'.format(data_type))
            stream = open(yml_dir, 'r')
            stats_train = yaml.load(stream, Loader=yaml.BaseLoader)
            mean = float(stats_train[dataset_name]['mean'])
            std = float(stats_train[dataset_name]['std'])
            self.stat = np.asarray([mean, std])

        # N(0, 1) normalization
        self.X = scaler.fit_transform(self.X)
        if not testing:
            self.Y = scaler.fit_transform(self.Y.reshape(-1, 1))

        self.input_dim = len(self.X[0])
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        # idx = 5

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

        return rtn_dict

    def collate_batch(self, batch):
        rtn_dict = {}
        for k, _ in batch[0].items():
            if k in ["input", "target"]:
                rtn_dict[k] = np.stack([sample[k] for sample in batch], axis=0)
        rtn_dict["stat"] = self.stat

        return rtn_dict

def dataset_split(dataset, fname, tar_split=-1, split_ratio=0.1, sort_by_target=False):
    X = dataset[:, :tar_split]
    Y = dataset[:, tar_split]
    # Y = dataset[:, tar_split:] #Currently we only use one target feature for all datasets. This line is for multi targets and is commented

    if sort_by_target:
        sorted_ids = np.argsort(Y)
        X = X[sorted_ids]
        Y = Y[sorted_ids]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, shuffle=False)
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=2020, shuffle=True)

    train = np.hstack([X_train, Y_train.reshape(len(X_train), -1)])
    test = np.hstack([X_test, Y_test.reshape(len(X_test), -1)])

    np.savetxt(fname + "_train.csv", train, delimiter=",", fmt="%10.6f")
    np.savetxt(fname + "_test.csv", test, delimiter=",", fmt="%10.6f")

    train = np.genfromtxt(fname + "_train.csv", delimiter=',')
    test = np.genfromtxt(fname + "_test.csv", delimiter=',')

    # mean_train = np.mean(train, axis=0)
    # std_train = np.std(train, axis=0)
    # mean_test = np.mean(test, axis=0)
    # std_test = np.std(test, axis=0)

if __name__ == "__main__":

    # Preprocessing normal data
    print("Processing UCI datasets")
    data_dirs = {}
    target_splits = {}
    data_dirs["boston"] = os.path.join("./..", "data", "boston")
    target_splits["boston"] = -1 # indicate which target feature to use
    data_dirs["wine"] = os.path.join("./..", "data", "wine")
    target_splits["wine"] = -1
    data_dirs["power_plant"] = os.path.join("./..", "data", "power_plant")
    target_splits["power_plant"] = -1
    data_dirs["concrete"] = os.path.join("./..", "data", "concrete")
    target_splits["concrete"] = -1

    data_dirs["energy"] = os.path.join("./..", "data", "energy")
    target_splits["energy"] = -2
    data_dirs["kin8nm"] = os.path.join("./..", "data", "kin8nm")
    target_splits["kin8nm"] = -1
    data_dirs["protein"] = os.path.join("./..", "data", "protein")
    target_splits["protein"] = -1

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

    # Preprocessing OOD data
    print('Processing OOD data')
    data_dirs = {}
    target_splits = {}
    data_dirs["boston"] = os.path.join("./..", "data_ood", "boston")
    target_splits["boston"] = -1
    data_dirs["wine"] = os.path.join("./..", "data_ood", "wine")
    target_splits["wine"] = -1
    data_dirs["power_plant"] = os.path.join("./..", "data_ood", "power_plant")
    target_splits["power_plant"] = -1
    data_dirs["concrete"] = os.path.join("./..", "data_ood", "concrete")
    target_splits["concrete"] = -1

    data_dirs["energy"] = os.path.join("./..", "data_ood", "energy")
    target_splits["energy"] = -2
    data_dirs["kin8nm"] = os.path.join("./..", "data_ood", "kin8nm")
    target_splits["kin8nm"] = -1
    data_dirs["protein"] = os.path.join("./..", "data_ood", "protein")
    target_splits["protein"] = -1

    data_files = {}
    datasets = {}
    for key, data_dir in data_dirs.items():
        os.makedirs(data_dir, exist_ok=True)
        data_files[key] = os.path.join(data_dir, key + ".csv")
        datasets[key] = np.genfromtxt(data_files[key], delimiter=',')[1:]
        mean = np.mean(datasets[key], axis=0)
        std = np.std(datasets[key], axis=0)
        dataset_split(datasets[key], os.path.join(data_dir, key), tar_split=target_splits[key], split_ratio=0.5, sort_by_target=True)

    naval_dir = os.path.join("./..", "data_ood", "naval")
    yacht_dir = os.path.join("./..", "data_ood", "yacht")
    naval = np.genfromtxt(os.path.join(naval_dir, "naval.txt"))
    yacht = np.genfromtxt(os.path.join(yacht_dir, "yacht.data"))
    dataset_split(naval, os.path.join(naval_dir, "naval"), tar_split=-2, split_ratio=0.5, sort_by_target=True)
    dataset_split(yacht, os.path.join(yacht_dir, "yacht"), split_ratio=0.5, sort_by_target=True)

    print("finished")



