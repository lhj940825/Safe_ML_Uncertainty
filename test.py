import torch
import torch.nn as nn
import sklearn.datasets as datasets
import pandas as pd
import os
import numpy as np

# m = nn.LogSoftmax(dim=0)
# n = nn.LogSoftmax(dim=1)
# o = nn.LogSoftmax()
# input = torch.randn(2, 3)
#
# print(input)
# print()
#
# output1 = m(input)
# output2 = n(input)
# output3 = o(input)
#
# output11 = o(input[:, 0])
# output22 = m(input[0, :])
# output33 = o(input[:, 0])
# output44 = o(input[:, 0])
# print(output1)
# print(output2)
# print(output3)
# print()
#
# print(output11)
# print(output22)
# print(output33)
# print(output44)

# data_dirs = {}
# target_splits = {}
# # data_dirs["energy"] = os.path.join("./..", "data", "energy")
# # target_splits["energy"] = -2
# # data_dirs["kin8nm"] = os.path.join("./..", "data", "kin8nm")
# # target_splits["kin8nm"] = -1
# data_dirs["naval"] = os.path.join(".", "data", "naval")
# data_dirs["yacht"] = os.path.join(".", "data", "yacht")
#
# data_files = {}
# datasets = {}
# for key, data_dir in data_dirs.items():
#     os.makedirs(data_dir, exist_ok=True)
#     data_files[key] = os.path.join(data_dir, key + ".csv")
#     datasets[key] = np.genfromtxt(data_files[key], delimiter=',')
#
# print()

a = np.ones((4,4))
b = np.array(range(1, 5))
c = a * b

print()