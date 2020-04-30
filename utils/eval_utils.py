import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.dataset import *

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

def eval(model, test_loader, cfg, tb_logger=None):
    NLL_list = None
    RMSE_list = None
    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            input, target = batch
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    samples = model(input).tolist()
                else:
                    model_output = model(input).tolist()
                    samples = np.append(samples, np.asarray(model_output), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])

            NLL = evaluate_with_NLL(mean, var, target.tolist())  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            #Log to tensorboard
            tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                RMSE_list = RMSE
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))

            tb_logger.flush()

    print("NLL min/max: ", min(NLL_list), max(NLL_list))

    print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

def evaluate_with_NLL(mean, var, label):
    epsilon = 1e-6

    var[var==0] =epsilon # replace where the value is zero to small number(epsilon) to prevent the operation being devided by zero
    var[var<epsilon] = epsilon
    NLL = np.log(var)*0.5 + np.divide(np.square(label-mean),(2*(var)))


    return NLL

def compute_mean_and_variance(samples, num_networks):
    """
    Compute the approximation of mean and variance for given input by MC dropout
    :param samples: for each input in batch, we draw (num_networks) samples from model [Batch_size, num_networks]
    :param num_networks: Dropout approximate the ensemble. num_networks means the number of times we draw samples from our dropout model.
    :return: approximated mean and variance
    """

    mean = np.mean(samples, axis=1) #shape(mean) = [batch_size, num_networks]
    # print('mean shape', np.shape(mean), mean)
    var =(np.sum(np.square(samples),axis=1))/num_networks - np.square(mean) # shape(var) = [Batch size, num_networks]

    return np.reshape(mean,(-1, 1)), np.reshape(var,(-1,1))

def evaluate_with_RMSE(mean, label):

    return np.sqrt(np.square(label - mean))