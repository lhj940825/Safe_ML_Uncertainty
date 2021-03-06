import torch
import torchvision
import os
from torch import optim
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn, model_fn_eval, eval, eval_with_training_dataset, eval_de
import utils.train_utils as tu
from utils.dataset import *
from utils.log_utils import create_tb_logger
from utils.utils import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # some cfgs, some cfg will be used in the future
    # TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 40
    cfg["ckpt_save_interval"] = 20
    cfg["batch_size"] = 100
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 10

    #TODO: Simplifiy and automate the process
    #Create directory for storing results
    output_dirs = {}

    output_dirs["boston"] = []
    output_dirs["concrete"] = []
    output_dirs["energy"] = []
    output_dirs["kin8nm"] = []
    output_dirs["naval"] = []
    output_dirs["power_plant"] = []
    output_dirs["protein"] = []
    output_dirs["wine"] = []
    output_dirs["yacht"] = []

    # output_dirs["year"] = []

    for key, sub_dirs in output_dirs.items():
        for idx in range(cfg["num_networks"]):
            sub_dirs.append(os.path.join('./output_de', key, 'model{}'.format(idx)))

    ckpt_dirs = {}
    for key, sub_dirs in output_dirs.items():
        ckpt_dirs[key] = []
        for idx, sub_dir in enumerate(sub_dirs):
            ckpt_dirs[key].append(os.path.join(sub_dir,'ckpts'))
            os.makedirs(ckpt_dirs[key][idx], exist_ok=True)

    data_dirs = {}
    for key, val in output_dirs.items():
        data_dirs[key] = os.path.join("./data", key)

    data_files = {}
    for key, _ in data_dirs.items():
        data_files[key] = ["{}_train.csv".format(key), "{}_eval.csv".format(key), "{}_test.csv".format(key)]

    train_datasets = {}
    train_loaders = {}
    eval_datasets = {}
    eval_loaders = {}

    print("Prepare training data")
    for key, fname in data_files.items():
        train_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[0]), testing=True)
        train_loaders[key] = torch.utils.data.DataLoader(train_datasets[key],
                                                         batch_size=cfg["batch_size"],
                                                         num_workers=0,
                                                         collate_fn=train_datasets[key].collate_batch,)
        eval_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[2]), testing=True)
        eval_loaders[key] = torch.utils.data.DataLoader(eval_datasets[key],
                                                        batch_size=cfg["batch_size"],
                                                        num_workers=0,
                                                        collate_fn=eval_datasets[key].collate_batch)

    #Prepare model
    print("Prepare model")
    from model.de_base import DE_base
    import random

    models = {}
    for key, dataset in train_datasets.items():
        models[key] = []
        for idx in range(cfg["num_networks"]):
            init_factor = random.uniform(0.0, 1.0)
            models[key].append(DE_base(dataset.input_dim, init_factor=init_factor))
            models[key][idx].cuda()

    # Testing
    print("Start testing")
    test_datasets = {}
    test_loaders = {}
    for key, fname in data_files.items():
        test_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[2]), testing=True)
        test_loaders[key] = torch.utils.data.DataLoader(test_datasets[key],
                                                        batch_size=cfg["batch_size"],
                                                        num_workers=0,
                                                        collate_fn=test_datasets[key].collate_batch)

    cur_ckpts = {}
    for key, ckpt_dir in ckpt_dirs.items():
        cur_ckpts[key] = []
        for idx, sub_dir in enumerate(ckpt_dir):
            cur_ckpts[key].append('{}.pth'.format(os.path.join(sub_dir, "ckpt_e{}".format(cfg['num_epochs']))))
            print("loading checkpoint {}".format(cur_ckpts[key][idx]))
            models[key][idx].load_state_dict(torch.load(cur_ckpts[key][idx])["model_state"])
            models[key][idx].eval()

    # Summarize the result into table and save it
    results = {}
    for key, model in models.items():
        eval_output = os.path.join('./output_de', key, 'eval')
        os.makedirs(eval_output, exist_ok=True)
        print("==================================Evaluating {}==========================================".format(key))
        result = eval_de(model, test_loader=test_loaders[key], cfg=cfg, output_dir=eval_output, title='test-' + key)
        results[key] = result

        # TODO below function is to generate figures for training dataset as requested by Joachim
        eval_de(model, train_loaders[key], cfg=cfg, output_dir=eval_output, title='train-' + key)
        print("Finished\n")

    dataset_list = []
    NLL_list = []
    NLL_without_v_Noise_list = []
    RMSE_list = []
    NLL_over_cap_cnt = []
    cap = 0

    for key, val in results.items():
        dataset_list.append(key)
        NLL_list.append(val[0][0])
        RMSE_list.append(val[0][1])
        NLL_over_cap_cnt.append(val[2][0])
        cap = val[2][1]
        NLL_without_v_Noise_list.append(val[3][0])

    err_df = pd.DataFrame(index=range(len(dataset_list)), columns=["Datasets", "RMSE", "NLL", "NLL_no_v_noise"])
    err_df["Datasets"] = pd.DataFrame(dataset_list)
    err_df["RMSE"] = pd.DataFrame(RMSE_list)
    err_df["NLL"] = pd.DataFrame(NLL_list)
    err_df["NLL_no_v_noise"] = pd.DataFrame(NLL_without_v_Noise_list)

    err_sum_dir = "./output_de/err_summary"
    os.makedirs(err_sum_dir, exist_ok=True)
    err_df.to_csv(os.path.join(err_sum_dir, "err_summary.csv"))

    plot_NLL_cap_cnt(dataset_list, NLL_over_cap_cnt, cap, err_sum_dir)

    # Finalizing
    print("Analysis finished\n")