import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def model_fn(model, data):
    rtn_dict = {}
    #unpack data
    input, target = data
    #Move data to GPU
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    pred = model(input)

    loss = model.loss_fn(pred, target)

    return loss

def eval_batch(model, data):
    loss = model_fn(model, data)