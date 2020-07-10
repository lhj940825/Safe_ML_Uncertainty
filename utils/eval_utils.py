import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from utils.dataset import *
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import *
from model import Wine_FC
# from utils.dataset import WineDataset
from sys import platform

# const variable
cap = 5

def model_fn_for_pu(model: torch.nn.Module, batch):
    """
    compute the loss(NLL) given batch.

    :param model:
    :param batch:
    :return: Computed loss given batch
    """

    #Move data to GPU
    input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
    target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
    target = target.reshape(-1, 1)

    pred = model(input)
    mean = pred[:, 0]
    output2 = pred[:, 1]
    #output2 = output2 + model.bias
    loss = model.loss_fn(target, mean, output2)

    # print("weights:")
    #
    # for weight in model.parameters():
    #     print(weight.data)

    return loss

def model_fn_eval_for_pu(model, eval_loader):
    model.eval()
    eval_loss = 0.0
    for it, batch in enumerate(eval_loader):

        # loss = model_fn(model, batch)

        # input, target = batch
        input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
        target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
        target = target.reshape(-1, 1)
        stat = batch["stat"] # stat = [Y_mean, Y_std]

        pred = model(input)
        # pred = stat[1] * pred + stat[0] # code for normalization new normalization

        mean = pred[:, 0]
        std = pred[:, 1]
        loss = model.loss_fn(target, mean, std)

        eval_loss += loss.item()
    return eval_loss / len(eval_loader)


def model_fn(model, batch):
    #unpack data
    # input, target = data
    # a = torch.mean(input, dim=0)
    # b = torch.std(input, dim=0)
    # c = torch.std(input[:, 0])
    # input, target, _ = data # code for normalization new normalization
    #Move data to GPU
    input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
    target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
    target = target.reshape(-1, 1)

    pred = model(input)

    loss = model.loss_fn(pred, target)

    return loss


def model_fn_eval(model, eval_loader, network_type='pu'):
    model.eval()
    eval_loss = 0.0
    mean_list = []
    var_list = []
    for it, batch in enumerate(eval_loader):

        input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
        target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
        target = target.reshape(-1, 1)
        stat = batch["stat"] # stat = [Y_mean, Y_std]

        pred = model(input)
        # if network_type == 'pu':
        #     mean = pred[:, 0]
        #     mean = stat[1] * mean + stat[0]  # denormalization to compute the mean
        #     mean = target - mean.view(-1, 1)
        #     mean = np.reshape(mean.cpu().data.numpy(), (-1, 1))
        #
        #     std = pred[:, 1]
        #     std = stat[1] * std  # denormalization to compute the std
        #
        #     var = np.reshape(torch.pow(std, 2).cpu().data.numpy(), (-1, 1))
        #
        #     mean_list.append(mean)
        #     var_list.append(var)

        # Normalized network output
        if network_type == 'pu':
            mean = pred[:, 0] # normalized mean
            target = (target - stat[0]) / stat[1] # normalized target
            mean = target - mean.view(-1, 1)
            mean = np.reshape(mean.cpu().data.numpy(), (-1, 1))

            std = pred[:, 1]
            var = np.reshape(torch.pow(std, 2).cpu().data.numpy(), (-1, 1)) # normalized variance

            mean_list.append(mean)
            var_list.append(var)

    mean_list = np.array(mean_list)
    var_list = np.array(var_list)

    mean_list = np.vstack(mean_list)
    var_list = np.vstack(var_list)

    return mean_list, var_list

        # loss = model.loss_fn(pred, target)
        # eval_loss += loss.item()
    # return eval_loss / len(eval_loader)

def eval(model, test_loader, cfg, output_dir, tb_logger=None, title=""):
    """
    Evaluation function for MC Dropout models

    :param model:
    :param test_loader:
    :param cfg:
    :param output_dir:
    :param tb_logger:
    :param title:
    :return:
    """
    NLL_list = None
    NLL_without_v_noise_list = None
    RMSE_list = None
    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)
    gt_M_distance_list = []
    sample_M_distance_list = []


    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    # samples = model(input).tolist()
                    out = stat[1] * model(input) + stat[0] # code for de-normalization
                    samples = out.tolist()
                else:
                    # model_output = model(input).tolist()
                    # samples = np.append(samples, np.asarray(model_output), axis=1)
                    out = stat[1] * model(input) + stat[0] # code for de-normalization
                    out = out.tolist()
                    samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])
            # mean = stat[1] * mean + stat[0]

            NLL, NLL_without_v_noise = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            sample = (stat[1]*model(input)+stat[0]).tolist()
            sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            sample_M_distance_list.extend(sample_M_distance)
            gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                NLL_without_v_noise_list = NLL_without_v_noise
                RMSE_list = RMSE
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                NLL_without_v_noise_list = np.append(NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

            tb_logger.flush()

    print("NLL min/max: ", min(NLL_list), max(NLL_list))

    print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])

    err_summary = np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                              [np.std(NLL_list), np.std(RMSE_list)],
                              [len(NLL_list[NLL_list>cap]), cap], # add the information of current cap-value and number of NLL values beyond such cap
                              [np.mean(NLL_without_v_noise_list), np.std(NLL_without_v_noise_list)]])

    err_list = np.append(err_list, err_summary, axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list,output_dir=output_dir,title=title)
    return err_summary


def ood_eval(pu_model, mc_model, test_loader, output_dir,cfg, tb_logger=None, title="", cur_epoch = None):
    """
    a function to perform the comparision between the evaluation of parametric uncertainty model and MC Dropout out model

    :param pu_model:
    :param mc_model:
    :param test_loader:
    :param output_dir:
    :param tb_logger:
    :param title:
    :return:
    """

    # lists for Parametric Uncertainty models
    pu_NLL_list = None
    pu_NLL_without_v_noise_list = None
    pu_RMSE_list = None
    pu_mean_list = None
    pu_variance_list = None
    pu_gt_list = None # to store all labels(ground truth)

    # lists for MC Dropout models
    mc_NLL_list = None
    mc_NLL_without_v_noise_list = None
    mc_RMSE_list = None
    mc_mean_list = None
    mc_variance_list = None
    mc_gt_list = None # to store all labels(ground truth)
    gt_M_distance_list = []
    sample_M_distance_list = []

    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            #TODO Compute the PU model output
            out = pu_model(input)
            mean = out[:,0]
            mean = stat[1]*mean+stat[0] #denormalization to compute the mean
            mean = np.reshape(mean.cpu().data.numpy(),(-1,1))

            std = stat[1]*out[:,1] #denormalization to compute the std
            #std = torch.exp(std)
            var = np.reshape(torch.pow(std,2).cpu().data.numpy(), (-1,1))

            NLL, NLL_without_v_noise  = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if pu_NLL_list is None:
                pu_NLL_list = NLL
                pu_NLL_without_v_noise_list = NLL_without_v_noise
                pu_RMSE_list = RMSE
                pu_mean_list = mean
                pu_variance_list = var
                pu_gt_list = target.tolist()
            else:
                pu_NLL_list = np.append(pu_NLL_list, np.squeeze(NLL))
                pu_NLL_without_v_noise_list = np.append(pu_NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                pu_RMSE_list = np.append(pu_RMSE_list, np.squeeze(RMSE))
                pu_mean_list = np.append(pu_mean_list, np.squeeze(mean))
                pu_variance_list = np.append(pu_variance_list, np.squeeze(var))
                pu_gt_list = np.append(pu_gt_list, np.squeeze(target.tolist()))

            #tb_logger.flush()
            #TODO Compute MCDropout output
            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    # samples = model(input).tolist()
                    out = stat[1] * mc_model(input) + stat[0] # code for de-normalization
                    samples = out.tolist()
                else:
                    # model_output = model(input).tolist()
                    # samples = np.append(samples, np.asarray(model_output), axis=1)
                    out = stat[1] * mc_model(input) + stat[0] # code for de-normalization
                    out = out.tolist()
                    samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])

            NLL, NLL_without_v_noise = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            sample = (stat[1]*mc_model(input)+stat[0]).tolist()
            sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            sample_M_distance_list.extend(sample_M_distance)
            gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if mc_NLL_list is None:
                mc_NLL_list = NLL
                mc_NLL_without_v_noise_list = NLL_without_v_noise
                mc_RMSE_list = RMSE
                mc_mean_list = mean
                mc_variance_list = var
                mc_gt_list = target.tolist()

            else:
                mc_NLL_list = np.append(mc_NLL_list, np.squeeze(NLL))
                mc_NLL_without_v_noise_list = np.append(mc_NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                mc_RMSE_list = np.append(mc_RMSE_list, np.squeeze(RMSE))
                mc_mean_list = np.append(mc_mean_list, np.squeeze(mean))
                mc_variance_list = np.append(mc_variance_list, np.squeeze(var))
                mc_gt_list = np.append(mc_gt_list, np.squeeze(target.tolist()))


    pu_err_summary = np.asarray([[np.mean(pu_NLL_list), np.mean(pu_RMSE_list)],
                                      [np.std(pu_NLL_list), np.std(pu_RMSE_list)],
                                      [len(pu_NLL_list[pu_NLL_list>cap]), cap], # add the information of current cap-value and number of NLL values beyond such cap
                                      [np.mean(pu_NLL_without_v_noise_list), np.std(pu_NLL_without_v_noise_list)]])

    mc_err_summary = np.asarray([[np.mean(mc_NLL_list), np.mean(mc_RMSE_list)],
                                 [np.std(mc_NLL_list), np.std(mc_RMSE_list)],
                                 [len(mc_NLL_list[mc_NLL_list>cap]), cap], # add the information of current cap-value and number of NLL values beyond such cap
                                 [np.mean(mc_NLL_without_v_noise_list), np.std(mc_NLL_without_v_noise_list)]])

    residual_error_and_std_plot_with_y_equal_abs_x_graph_for_pu_and_mc_of_ood(pu_gt_list - pu_mean_list, np.sqrt(pu_variance_list), mc_gt_list - mc_mean_list, np.sqrt(mc_variance_list), output_dir, title, y_axis_contraint=True, y_max=3, cur_epoch=cur_epoch)
    residual_error_and_std_plot_with_y_equal_abs_x_graph_for_pu_and_mc_of_ood(pu_gt_list - pu_mean_list, np.sqrt(pu_variance_list), mc_gt_list - mc_mean_list, np.sqrt(mc_variance_list), output_dir, title, y_axis_contraint=False, cur_epoch=cur_epoch)
    NLL_histogram_for_pu_and_mc_ood_dataset(pu_NLL_list,mc_NLL_list,output_dir, title, cur_epoch
                                            )
    return pu_err_summary, mc_err_summary


def pu_eval(model, test_loader, cfg, output_dir, tb_logger=None, title=""):
    """
    a function to evaluate the parametric uncertainty model over test dataset.

    :param model:
    :param test_loader:
    :param cfg:
    :param output_dir:
    :param tb_logger:
    :param title:
    :return:
    """

    NLL_list = None
    NLL_without_v_noise_list = None
    RMSE_list = None
    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)
    #gt_M_distance_list = []
    #sample_M_distance_list = []

    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            out = model(input)
            mean = out[:,0]
            mean = stat[1]*mean+stat[0] #denormalization to compute the mean
            mean = np.reshape(mean.cpu().data.numpy(),(-1,1))

            std = stat[1]*out[:,1] #denormalization to compute the std
            #std = torch.exp(std)
            var = np.reshape(torch.pow(std,2).cpu().data.numpy(), (-1,1))

            NLL, NLL_without_v_noise  = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            #sample = (stat[1]*model(input)+stat[0]).tolist()
            #sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            #sample_M_distance_list.extend(sample_M_distance)
            #gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                NLL_without_v_noise_list = NLL_without_v_noise
                RMSE_list = RMSE
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                NLL_without_v_noise_list = np.append(NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

            tb_logger.flush()

    print("NLL min/max: ", min(NLL_list), max(NLL_list))

    print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])

    err_summary = np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                              [np.std(NLL_list), np.std(RMSE_list)],
                              [len(NLL_list[NLL_list>cap]), cap], # add the information of current cap-value and number of NLL values beyond such cap
                              [np.mean(NLL_without_v_noise_list), np.std(NLL_without_v_noise_list)]])


    err_list = np.append(err_list, err_summary, axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    #plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    #plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list,output_dir=output_dir,title=title)
    return err_summary


def eval_batch(model, data):
    loss = model_fn(model, data)



def eval_with_training_dataset(model, train_loader, cfg, output_dir, tb_logger=None, title=""):
    NLL_list = None
    NLL_without_v_noise_list = None
    RMSE_list = None
    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)
    gt_M_distance_list = []
    sample_M_distance_list = []

    if platform=='win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(train_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    # samples = model(input).tolist()
                    out = stat[1] * model(input) + stat[0] # code for de-normalization
                    samples = out.tolist()
                else:
                    # model_output = model(input).tolist()
                    # samples = np.append(samples, np.asarray(model_output), axis=1)
                    out = stat[1] * model(input) + stat[0] # code for de-normalization
                    out = out.tolist()
                    samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])
            # mean = stat[1] * mean + stat[0]

            NLL, NLL_without_v_noise  = evaluate_with_NLL(mean, var, target.tolist(),dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            sample = (stat[1]*model(input)+stat[0]).tolist()
            sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            sample_M_distance_list.extend(sample_M_distance)
            gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                NLL_without_v_noise_list = NLL_without_v_noise
                RMSE_list = RMSE
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                NLL_without_v_noise_list = np.append(NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

            #tb_logger.flush()

    #print("NLL min/max: ", min(NLL_list), max(NLL_list))

    #print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    #print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])

    err_summary = np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                              [np.std(NLL_list), np.std(RMSE_list)],
                              [len(NLL_list[NLL_list>cap]), cap]]) # add the information of current cap-value and number of NLL values beyond such cap

    err_list = np.append(err_list, err_summary, axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list,output_dir=output_dir,title=title)
    return err_summary

def eval_de(models, test_loader, cfg, output_dir, tb_logger=None, title=""):
    NLL_list = None
    RMSE_list = None
    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)
    gt_M_distance_list = []
    sample_M_distance_list = []
    norm_y = []



    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(test_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    # samples = model(input).tolist()
                    out = stat[1] * models[i](input) + stat[0] # code for de-normalization
                    samples = out.tolist()
                else:
                    # model_output = model(input).tolist()
                    # samples = np.append(samples, np.asarray(model_output), axis=1)
                    out = stat[1] * models[i](input) + stat[0] # code for de-normalization
                    out = out.tolist()
                    samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])
            norm_y.append(((samples - mean) / np.sqrt(var)).reshape(-1))

            NLL, NLL_without_v_noise = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            sample = (stat[1] * models[i](input) + stat[0]).tolist()
            sample_M_distance, gt_M_distance = assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            sample_M_distance_list.extend(sample_M_distance)
            gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            # tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if NLL_list is None:
                NLL_list = NLL
                NLL_without_v_noise_list = NLL_without_v_noise
                RMSE_list = RMSE
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()
            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                NLL_without_v_noise_list = np.append(NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

            # tb_logger.flush()

    norm_y = np.hstack(norm_y).reshape(-1)

    print("NLL min/max: ", min(NLL_list), max(NLL_list))

    print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])

    err_summary = np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                              [np.std(NLL_list), np.std(RMSE_list)],
                              [len(NLL_list[NLL_list > cap]), cap],
                              # add the information of current cap-value and number of NLL values beyond such cap
                              [np.mean(NLL_without_v_noise_list), np.std(NLL_without_v_noise_list)]])


    err_list = np.append(err_list, err_summary, axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_histograms(norm_y, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list,output_dir=output_dir,title=title)
    return err_summary

def pu_eval_with_training_dataset(model, train_loader, cfg, output_dir, tb_logger=None, title=""):
    """
    a function to evaluate the parametric uncertainty model over training dataset

    :param model:
    :param train_loader:
    :param cfg:
    :param output_dir:
    :param tb_logger:
    :param title:
    :return:
    """
    NLL_list = None
    NLL_without_v_noise_list = None
    RMSE_list = None
    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)
    #gt_M_distance_list = []
    #sample_M_distance_list = []

    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(train_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            model.eval()
            out = model(input)
            mean = out[:,0]
            mean = stat[1]*mean+stat[0] #denormalization to compute the mean
            mean = np.reshape(mean.cpu().data.numpy(),(-1,1))

            std = stat[1]*out[:,1] #denormalization to compute the std
            #std = torch.exp(std)
            var = np.reshape(torch.pow(std,2).cpu().data.numpy(), (-1,1))

            NLL, NLL_without_v_noise  = evaluate_with_NLL(mean, var, target.tolist(),dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            #sample = (stat[1]*model(input)+stat[0]).tolist()
            #sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            #sample_M_distance_list.extend(sample_M_distance)
            #gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)
            if NLL_list is None:
                NLL_list = NLL
                NLL_without_v_noise_list = NLL_without_v_noise
                RMSE_list = RMSE
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()

            else:
                NLL_list = np.append(NLL_list, np.squeeze(NLL))
                NLL_without_v_noise_list = np.append(NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                RMSE_list = np.append(RMSE_list, np.squeeze(RMSE))
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

            #tb_logger.flush()

    #print("NLL min/max: ", min(NLL_list), max(NLL_list))

    #print('NLL result mean:{}, standard deviation:{}'.format(np.mean(NLL_list), np.std(NLL_list)))
    #print('RMSE result mean:{}, standard deviation:{}'.format(np.mean(RMSE_list), np.std(RMSE_list)))

    err_list = np.hstack([NLL_list.reshape(-1, 1), RMSE_list.reshape(-1, 1)])

    err_summary = np.asarray([[np.mean(NLL_list), np.mean(RMSE_list)],
                              [np.std(NLL_list), np.std(RMSE_list)],
                              [len(NLL_list[NLL_list>cap]), cap]]) # add the information of current cap-value and number of NLL values beyond such cap

    err_list = np.append(err_list, err_summary, axis=0)

    np.savetxt(os.path.join(output_dir, title + "_error_list.csv"), err_list, delimiter=",")

    # err_df = pd.DataFrame(err_list)
    # err_df.columns = ['NLL', "RMSE"]
    # err_df.append(pd.DataFrame(np.asarray([[np.mean(NLL_list), np.std(NLL_list)],
    #                                        [np.mean(RMSE_list), np.std(RMSE_list)]])))
    # err_df.to_csv(os.path.join(output_dir, title + "_error_list.csv"))

    plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter(NLL_list, RMSE_list, output_dir, title=title)
    plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    #plot_Mahalanobis_distance(sample_M_distance_list,gt_M_distance_list, output_dir=output_dir, title=title)
    #plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list,output_dir=output_dir,title=title)
    return err_summary

def pu_eval_residualError_and_std_with_particular_epoch(model, train_loader, output_dir, title=""):

    mean_list = None
    variance_list = None
    gt_list = None # to store all labels(ground truth)

    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():
        for cur_it, batch in enumerate(train_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            model.eval()
            out = model(input)
            mean = out[:,0]
            mean = stat[1]*mean+stat[0] #denormalization to compute the mean
            mean = np.reshape(mean.cpu().data.numpy(),(-1,1))

            std = stat[1]*out[:,1] #denormalization to compute the std
            #std = torch.exp(std)
            var = np.reshape(torch.pow(std,2).cpu().data.numpy(), (-1,1))

            if mean_list is None:
                mean_list = mean
                variance_list = var
                gt_list = target.tolist()

            else:
                mean_list = np.append(mean_list, np.squeeze(mean))
                variance_list = np.append(variance_list, np.squeeze(var))
                gt_list = np.append(gt_list, np.squeeze(target.tolist()))

    #plot_scatter2(gt_list, mean_list, variance_list, output_dir, title)
    plot_scatter2_and_NLL_histogram_variants(gt_list, mean_list, variance_list, output_dir, title)

def ood_eval_with_training_datset(pu_model, mc_model, train_loader, output_dir,cfg, tb_logger=None, title="", cur_epoch = None):

    # lists for Parametric Uncertainty models
    pu_NLL_list = None
    pu_NLL_without_v_noise_list = None
    pu_RMSE_list = None
    pu_mean_list = None
    pu_variance_list = None
    pu_gt_list = None # to store all labels(ground truth)

    # lists for MC Dropout models
    mc_NLL_list = None
    mc_NLL_without_v_noise_list = None
    mc_RMSE_list = None
    mc_mean_list = None
    mc_variance_list = None
    mc_gt_list = None # to store all labels(ground truth)
    #gt_M_distance_list = []
    #sample_M_distance_list = []

    if platform == 'win32':
        dataset_name = output_dir.split("\\")[1]
    else:
        dataset_name = output_dir.split("/")[2]

    with torch.no_grad():

        for cur_it, batch in enumerate(train_loader):
            # input, target = batch
            input = torch.from_numpy(batch["input"]).cuda(non_blocking=True).float()
            target = torch.from_numpy(batch["target"]).cuda(non_blocking=True).float()
            target = target.reshape(-1, 1)
            stat = batch["stat"]

            #TODO prior is no more used to compute v-noise
            #prior = batch["prior"]
            # stat = stat.cuda(non_blocking=True)[0] # code for normalization new normalization
            #v_noise = prior[1] / (prior[0] - 1) * stat[1] ** 2

            #TODO Compute the PU model output
            out = pu_model(input)
            mean = out[:,0]
            mean = stat[1]*mean+stat[0] #denormalization to compute the mean
            mean = np.reshape(mean.cpu().data.numpy(),(-1,1))

            std = stat[1]*out[:,1] #denormalization to compute the std
            #std = torch.exp(std)
            var = np.reshape(torch.pow(std,2).cpu().data.numpy(), (-1,1))

            NLL, NLL_without_v_noise  = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if pu_NLL_list is None:
                pu_NLL_list = NLL
                pu_NLL_without_v_noise_list = NLL_without_v_noise
                pu_RMSE_list = RMSE
                pu_mean_list = mean
                pu_variance_list = var
                pu_gt_list = target.tolist()
            else:
                pu_NLL_list = np.append(pu_NLL_list, np.squeeze(NLL))
                pu_NLL_without_v_noise_list = np.append(pu_NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                pu_RMSE_list = np.append(pu_RMSE_list, np.squeeze(RMSE))
                pu_mean_list = np.append(pu_mean_list, np.squeeze(mean))
                pu_variance_list = np.append(pu_variance_list, np.squeeze(var))
                pu_gt_list = np.append(pu_gt_list, np.squeeze(target.tolist()))

            #tb_logger.flush()
            #TODO Compute MCDropout output
            samples = None
            for i in range(cfg["num_networks"]):  # By MC dropout, samples the network output several times(=num_networks) given the same input in order to compute the mean and variance for such given input
                if samples is None:
                    # samples = model(input).tolist()
                    out = stat[1] * mc_model(input) + stat[0] # code for de-normalization
                    samples = out.tolist()
                else:
                    # model_output = model(input).tolist()
                    # samples = np.append(samples, np.asarray(model_output), axis=1)
                    out = stat[1] * mc_model(input) + stat[0] # code for de-normalization
                    out = out.tolist()
                    samples = np.append(samples, np.asarray(out), axis=1)

            mean, var = compute_mean_and_variance(samples, num_networks=cfg["num_networks"])

            NLL, NLL_without_v_noise = evaluate_with_NLL(mean, var, target.tolist(), dataset_name)  # compute the Negative log likelihood with mean, var, target value(label)
            RMSE = evaluate_with_RMSE(mean, target.tolist())

            #sample = (stat[1]*mc_model(input)+stat[0]).tolist()
            #sample_M_distance, gt_M_distance =assess_uncertainty_realism(gt_label=target.tolist(),sample=sample, mean=mean, var=var)
            #sample_M_distance_list.extend(sample_M_distance)
            #gt_M_distance_list.extend(gt_M_distance)

            #Log to tensorboard
            #tb_logger.add_scalar('Loss/test_loss', np.average(NLL), cur_it)

            if mc_NLL_list is None:
                mc_NLL_list = NLL
                mc_NLL_without_v_noise_list = NLL_without_v_noise
                mc_RMSE_list = RMSE
                mc_mean_list = mean
                mc_variance_list = var
                mc_gt_list = target.tolist()

            else:
                mc_NLL_list = np.append(mc_NLL_list, np.squeeze(NLL))
                mc_NLL_without_v_noise_list = np.append(mc_NLL_without_v_noise_list, np.squeeze(NLL_without_v_noise))
                mc_RMSE_list = np.append(mc_RMSE_list, np.squeeze(RMSE))
                mc_mean_list = np.append(mc_mean_list, np.squeeze(mean))
                mc_variance_list = np.append(mc_variance_list, np.squeeze(var))
                mc_gt_list = np.append(mc_gt_list, np.squeeze(target.tolist()))


    residual_error_and_std_plot_with_y_equal_abs_x_graph_for_pu_and_mc_of_ood(pu_gt_list - pu_mean_list, np.sqrt(pu_variance_list), mc_gt_list - mc_mean_list, np.sqrt(mc_variance_list), output_dir, title, y_axis_contraint=True, y_max=3, cur_epoch=cur_epoch)
    residual_error_and_std_plot_with_y_equal_abs_x_graph_for_pu_and_mc_of_ood(pu_gt_list - pu_mean_list, np.sqrt(pu_variance_list), mc_gt_list - mc_mean_list, np.sqrt(mc_variance_list), output_dir, title, y_axis_contraint=False, cur_epoch=cur_epoch)
    #NLL_histogram_for_pu_and_mc_ood_dataset(pu_NLL_list,mc_NLL_list,output_dir, title, cur_epoch

    return







def compute_test_loss(model: nn.Module, test_set_dir, batch_size, num_worker, device):
    """
    A function to compute the loss of model in training phase given the test dataset.

    :param model:
    :param test_set_dir:
    :param batch_size:
    :param dropout_prop:
    :param num_worker:
    :param input_dim:
    :param device: GPU or CPU
    :return:
    """
    print('Start Evaluating with MC Dropout')
    test_dataset = WineDataset(data_dir=test_set_dir,transform=None)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True, num_workers=num_worker)

    model = model.float()
    model = model.to(device)

    model.train()

    test_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            input, label = batch

            # preprocessing inputs and labels in order to prevent some runtime error( i.e tensor dim or datatype not matching)
            input = input.to(device)
            input = input.float()
            label = label.to(device)
            label = label.float()
            label = label.view(-1,1) #Tensor reshape

            model_output = model(input)
            MSELoss = nn.MSELoss()
            loss = MSELoss(model_output, label)
            test_loss += loss

    test_loss = test_loss/len(test_dataloader)
    print("Test loss: ",test_loss)

    return test_loss


def evaluate_with_NLL(mean, var, label, dataset_name, v_noise=1):
    import yaml
    #TODO chosen values for v_noise
    # 1e-3: yacht(2.193789627	1.52338537)
    # 1e-2: boston(2.469, 2.739), energy(1.67, 1.71)
    # 1e-1: wine(0.51, 1.49), kin8nm(0.106, -0.06)
    # 1: power plant(5.46, 2.91), concrete(6.29, 2.945), year(8.806,2.89), naval(0.027,0.1515), protein

    # load the std_target_train from mean_std.yaml file
    yml_dir = os.path.join(os.getcwd(), 'configs/mean_std.yml')
    stream = open(yml_dir, 'r')
    data = yaml.load(stream,Loader=yaml.BaseLoader)
    std_target_train = float(data[dataset_name]['std'])
    v_noise = float(data[dataset_name]['v_noise'])

    # compute NLL without applying v-noise
    var_without_v_noise = np.copy(var)
    var_without_v_noise[var_without_v_noise ==0] = 1e-6 #hadnling variance of 0
    NLL_without_v_noise = np.log(var_without_v_noise)*0.5 + np.divide(np.square(label-mean), (2*(var_without_v_noise)))

    # compute variance with v-noise
    var = var + (std_target_train**2)*v_noise

    a = np.log(var)*0.5
    b = np.divide(np.square(label-mean), (2*(var)))

    # compute NLL with v_noise
    NLL = a + b

    return NLL, NLL_without_v_noise


def compute_mean_and_variance(samples, num_networks):
    """
    Compute the approximation of mean and variance for given input by MC dropout
    :param samples: for each input in batch, we draw (num_networks) samples from model [Batch_size, num_networks]
    :param num_networks: Dropout approximate the ensemble. num_networks means the number of times we draw samples from our dropout model.
    :return: approximated mean and variance
    """

    mean = np.mean(samples, axis=1) #shape(mean) = [batch_size, num_networks]
    # print('mean shape', np.shape(mean), mean)
    
    var = np.square(np.std(samples, axis=1))

    return np.reshape(mean,(-1, 1)), np.reshape(var,(-1,1))

def evaluate_with_RMSE(mean, label):

    return np.sqrt(np.square(label - mean))

def assess_uncertainty_realism(gt_label, sample, mean, var):
    """
    compute the 1-dimension Mahalanobis distance given the ground truth label and sample drawn from the model

    :param gt_label: ground truth label
    :param sample: single sample drawn from MC dropout model
    :param mean: sample mean of 50 samples
    :param var: sample variance of 50 samples
    :return: list of computed sample Mahalanobis distance and gt Mahalanobis distance
    """
    epsilon = 1e-6
    var[var == 0] = epsilon
    sample_M_distance = np.divide(np.square(sample-mean),var) # computed Mahalanobis distance given sample, but since the model output is 1D, we compute 1D version of Mahalanobis distance
    gt_M_distance = np.divide(np.square(gt_label-mean),var) # computed Mahalanobis distance given ground truth label

    return (np.squeeze(sample_M_distance).tolist(), np.squeeze(gt_M_distance).tolist())