import torch
import torchvision
import os
from torch import optim
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn, model_fn_eval, eval, eval_with_training_dataset
import utils.train_utils as tu
from utils.dataset import *
from utils.log_utils import create_tb_logger
from utils.utils import *


if __name__ == "__main__":
    #TODO: Simplifiy and automate the process
    #Create directory for storing results
    output_dirs = {}
    output_dirs["boston"] = os.path.join("./", "output", "boston")
    output_dirs["wine"] = os.path.join("./", "output", "wine")
    output_dirs["power_plant"] = os.path.join("./", "output", "power_plant")
    output_dirs["concrete"] = os.path.join("./", "output", "concrete")
    output_dirs["energy"] = os.path.join("./", "output", "energy")
    output_dirs["kin8nm"] = os.path.join("./", "output", "kin8nm")

    ckpt_dirs = {}
    for key, val in output_dirs.items():
        os.makedirs(val, exist_ok=True)
        ckpt_dirs[key] = os.path.join(val, 'ckpts')
        os.makedirs(ckpt_dirs[key], exist_ok=True)

    #some cfgs, some cfg will be used in the future
    #TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 40
    cfg["ckpt_save_interval"] = 20
    cfg["batch_size"] = 100
    cfg["pdrop"] = 0.1
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 50

    data_dirs = {}
    for key, val in output_dirs.items():
        data_dirs[key] = os.path.join("./data", key)

    data_files = {}
    data_files["boston"] = ["boston_train.csv", "boston_test.csv"]
    data_files["wine"] = ["train_winequality-red.csv", "test_winequality-red.csv"]
    data_files["power_plant"] = ["pp_train.csv", "pp_test.csv"]
    data_files["concrete"] = ["concrete_train.csv", "concrete_test.csv"]
    data_files["energy"] = ["energy_train.csv", "energy_test.csv"]
    data_files["kin8nm"] = ["kin8nm_train.csv", "kin8nm_test.csv"]

    train_datasets = {}
    train_loaders = {}
    eval_datasets = {}
    eval_loaders = {}

    for key, fname in data_files.items():
        train_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[0]))
        train_loaders[key] = torch.utils.data.DataLoader(train_datasets[key], batch_size=cfg["batch_size"], num_workers=0)
        eval_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[1]), testing=True)
        eval_loaders[key] = torch.utils.data.DataLoader(eval_datasets[key], batch_size=cfg["batch_size"], num_workers=0)

    # dataiter = iter(train_loader_bos)
    # data, target = dataiter.next()

    #Prepare model
    print("Prepare model")
    from model.fc import FC

    models = {}
    for key, dataset in train_datasets.items():
        models[key] = FC(dataset.input_dim, cfg["pdrop"])
        models[key].cuda()

    #Prepare training
    print("Prepare training")
    optimizers = {}
    for key, model in models.items():
        optimizers[key] = optim.Adam(model.parameters(), lr=tu.lr_scheduler())

    #Define starting iteration/epochs.
    #Will use checkpoints in the future when running on clusters
    # if cfg["ckpt"] is not None:
    #     starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=cfg["ckpt"])
    # elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
    #     starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    # else:
    #     starting_iteration, starting_epoch = 0, 0

    starting_iteration, starting_epoch = 0, 0

    #Logging
    tb_loggers = {}
    for key, val in output_dirs.items():
        tb_loggers[key] = create_tb_logger(val)

    #Training
    print("Start training")

    trainers = {}

    for key, model in models.items():
        print("*******************************Training {}*******************************\n".format(key))
        trainers[key] = Trainer(model=model,
                              model_fn=model_fn,
                              model_fn_eval=model_fn_eval,
                              optimizer=optimizers[key],
                              ckpt_dir=ckpt_dirs[key],
                              grad_norm_clip=cfg["grad_norm_clip"],
                              tb_logger=tb_loggers[key])

        trainers[key].train(num_epochs=cfg["num_epochs"],
                         train_loader=train_loaders[key],
                         eval_loader=eval_loaders[key],
                         ckpt_save_interval=cfg["ckpt_save_interval"],
                         starting_iteration=starting_iteration,
                         starting_epoch=starting_epoch)

        draw_loss_trend_figure(key, trainers[key].train_loss, trainers[key].eval_loss, len(trainers[key].train_loss), output_dirs[key])
        print("*******************************Finished training {}*******************************\n".format(key))


    #Testing
    print("Start testing")
    # Currently the test dataset is the same as training set. TODO: K-Flod cross validation
    test_datasets = {}
    test_loaders = {}
    for key, fname in data_files.items():
        test_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[1]), testing=True)
        test_loaders[key] = torch.utils.data.DataLoader(test_datasets[key], batch_size=cfg["batch_size"], num_workers=0)

    cur_ckpts = {}
    for key, ckpt_dir in ckpt_dirs.items():
        # cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(trainers[key]._epoch + 1)))
        # print("loading checkpoint ckpt_e{}".format(trainers[key]._epoch + 1))
        cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(40)))
        print("loading checkpoint ckpt_e{}".format(40))
        models[key].load_state_dict(torch.load(cur_ckpts[key])["model_state"])
        models[key].train()

    # Summarize the result into table and save it
    results = {}
    for key, model in models.items():
        print("==================================Evaluating {}==========================================".format(key))
        result = eval(model, test_loader=test_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='test-'+key)
        results[key] = result

        #TODO below function is to generate figures for training dataset as requested by Joachim
        eval_with_training_dataset(model, train_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='train-'+key)
        print("Finished\n")

    dataset_list = []
    NLL_list = []
    RMSE_list = []
    for key, val in results.items():
        dataset_list.append(key)
        NLL_list.append(val[0][0])
        RMSE_list.append(val[0][1])

    err_df = pd.DataFrame(index=range(len(dataset_list)), columns=["Datasets", "RMSE", "NLL"])
    # a = pd.DataFrame(dataset_list)
    err_df["Datasets"] = pd.DataFrame(dataset_list)
    err_df["RMSE"] = pd.DataFrame(RMSE_list)
    err_df["NLL"] = pd.DataFrame(NLL_list)

    err_sum_dir = "./output/err_summary"
    os.makedirs(err_sum_dir, exist_ok=True)
    err_df.to_csv(os.path.join(err_sum_dir, "err_summary.csv"))

    #Finalizing
    print("Analysis finished\n")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future