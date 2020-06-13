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
    # output_dirs["boston"] = []
    # output_dirs["wine"] = []
    # output_dirs["power_plant"] = []
    output_dirs["concrete"] = []
    output_dirs["energy"] = []
    # output_dirs["kin8nm"] = []
    # output_dirs["naval"] = []
    # output_dirs["yacht"] = []
    # output_dirs["protein"] = []
    # output_dirs["year"] = []

    for key, sub_dirs in output_dirs.items():
        for idx in range(cfg["num_networks"]):
            sub_dir = os.path.join('./output_de', key, 'model{}'.format(idx))
            os.makedirs(sub_dir, exist_ok=True)
            sub_dirs.append(sub_dir)

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
        train_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[0]))
        train_loaders[key] = torch.utils.data.DataLoader(train_datasets[key],
                                                         batch_size=cfg["batch_size"],
                                                         num_workers=0,
                                                         collate_fn=train_datasets[key].collate_batch)
        eval_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[1]), testing=True)
        eval_loaders[key] = torch.utils.data.DataLoader(eval_datasets[key],
                                                        batch_size=cfg["batch_size"],
                                                        num_workers=0,
                                                        collate_fn=eval_datasets[key].collate_batch)

    # dataiter = iter(train_loader_bos)
    # data, target = dataiter.next()

    #Prepare model
    print("Prepare model")
    from model.de_base import DE_base
    import random

    models = {}
    for key, dataset in train_datasets.items():
        models[key] = []
        for idx in range(cfg["num_networks"]):
            models[key].append(DE_base(dataset.input_dim, init_factor=random.uniform(0.0, 1.0)))
            models[key][idx].cuda()

    #Prepare training
    print("Prepare training")
    optimizers = {}
    for key, model in models.items():
        optimizers[key] = []
        for base_net in model:
            optimizers[key].append(optim.Adam(base_net.parameters(), lr=tu.lr_scheduler()))

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
    # tb_loggers = {}
    # for key, val in output_dirs.items():
    #     tb_loggers[key] = create_tb_logger(val)

    #Training
    print("Start training")

    trainers = {}

    for key, model in models.items():
        print("*******************************Training {}*******************************\n".format(key))
        trainers[key] = []
        fig, axes = plt.subplots(5, 2)
        fig.suptitle("Loss figure {}".format(key))
        fig.set_size_inches(20, 25)

        for idx, base_net in enumerate(model):
            trainers[key].append(Trainer(model=base_net,
                                         model_fn=model_fn,
                                         model_fn_eval=model_fn_eval,
                                         optimizer=optimizers[key][idx],
                                         ckpt_dir=ckpt_dirs[key][idx],
                                         grad_norm_clip=cfg["grad_norm_clip"]))

            trainers[key][idx].train(num_epochs=cfg["num_epochs"],
                                     train_loader=train_loaders[key],
                                     # eval_loader=eval_loaders[key],
                                     eval_loader=None,
                                     ckpt_save_interval=cfg["ckpt_save_interval"],
                                     starting_iteration=starting_iteration,
                                     starting_epoch=starting_epoch)

        # draw_loss_trend_figure(key, len(trainers[key].train_loss), trainers[key].train_loss, trainers[key].eval_loss, output_dirs[key])
        #     draw_loss_trend_figure("Base_model{}_{}".format(idx, key), len(trainers[key][idx].train_loss), trainers[key][idx].train_loss, output_dir=output_dirs[key][idx])

            #Draw a all-in-one figure
            plot_loss(axes[idx // 2, idx % 2], "Base_model{}_{}".format(idx, key),
                      len(trainers[key][idx].train_loss),
                      trainers[key][idx].train_loss,
                      output_dir=output_dirs[key][idx])

        plt.savefig(os.path.join("./output_de", key, 'Loss_fig_all_in_one_{}.png'.format(key)))
        plt.show()
        print("*******************************Finished training {}*******************************\n".format(key))

    #Finalizing
    print("Training finished\n")