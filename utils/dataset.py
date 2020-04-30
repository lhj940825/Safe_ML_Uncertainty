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

def split_dataset(data_dir, train_set_dir, test_set_dir):

    if not os.path.exists(data_dir):
        raise FileNotFoundError('{} file is not found. need to download and place the file in the mentioned directory'.format(wine_data_dir))

    else: # when the dataset file 'winequality-white.csv' exists
        if not os.path.exists(train_set_dir): # when the dataset file exists but not has been splitted.
            wind_dataset = pd.read_csv(data_dir, sep=',')
            X = wind_dataset.drop(labels= 'quality', axis = 1)

            X = (X-X.min())/(X.max() - X.min()) # min-max normalization
            Y = wind_dataset['quality']

            train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.3, stratify=Y)


            df_train = pd.concat([train_x, train_y], axis=1) # concatanate training set with corresponding labels
            df_test = pd.concat([test_x, test_y], axis=1) # concatanate test set with corresponding labels

            # write splitted dataset to csv
            df_train.to_csv(train_set_dir, index=False)
            df_test.to_csv(test_set_dir, index=False)

class BostonDataset(Dataset):
    def __init__(self, data_path, transform=None):
        data = np.genfromtxt(data_path, delimiter=",") # load dataset
        scaler = StandardScaler()
        self.X = data[:, :-1]
        # self.X = (self.X - np.min(self.X)) / (np.max(self.X) - np.min(self.X))
        self.X = scaler.fit_transform(self.X)
        self.Y = data[:, -1]
        self.input_dim = len(self.X[0])

        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx])
        y = np.asarray(self.Y[idx])

        if self.transform:
            x = self.transform(x)

        a = torch.tensor(x, dtype=torch.float)
        b = torch.tensor(y, dtype=torch.float).view(-1)

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float).view(-1) #return in form of tensor

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


if __name__ == "__main__":
    # data_dir = os.path.join("./..", "data", "boston_housing")
    # os.makedirs(data_dir, exist_ok=True)
    # boston = datasets.load_boston()
    # X = boston.data
    # Y = boston.target
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2020, shuffle=True)
    # boston_train = np.append(X_train, Y_train.reshape(-1, 1), axis=1)
    # boston_test = np.append(X_test, Y_test.reshape(-1, 1), axis=1)
    #
    # np.savetxt(os.path.join(data_dir, "boston_train.csv"), boston_train, delimiter=",")
    # np.savetxt(os.path.join(data_dir, "boston_test.csv"), boston_test, delimiter=",")
    data_dir = os.path.join("./..", "data", "wine")
    wine_train = np.genfromtxt(os.path.join(data_dir, 'train_winequality-red.csv'), delimiter=',')[1:]
    wine_test = np.genfromtxt(os.path.join(data_dir, 'test_winequality-red.csv'), delimiter=',')[1:]
    print("finished")
