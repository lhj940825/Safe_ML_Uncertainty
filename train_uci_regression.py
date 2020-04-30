import torch
import torchvision
from torch import optim
import os
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn, eval
import utils.train_utils as tu
from utils.dataset import *
from utils.log_utils import create_tb_logger
import matplotlib.pyplot as plt

if __name__ == "__main__":

    #Create directory for storing results
    output_dir = os.path.join("./", "output", "uci_regression")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    #some cfgs, some cfg will be used in the future
    #TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 40
    cfg["ckpt_save_interval"] = 5
    cfg["batch_size"] = 100
    cfg["pdrop"] = 0.1
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 50

    # data_dir = os.path.join("./data", "boston_housing")
    data_dir = os.path.join("./data", "wine")

    #Prepare data Boston Housing coverted to pytorch dataset
    # bos_train = BostonDataset(os.path.join(data_dir, "boston_train.csv"))
    # train_loader_bos = torch.utils.data.DataLoader(bos_train, batch_size=cfg["batch_size"], num_workers=2)
    # trans = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Normalize(mean=[0.0, 0.0, 0.0],
    #                                                  std=[1.0, 1.0, 1.0])
    # ])

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.0],
                                                     std=[1.0])
                                ])

    wine_train = WineDataset(os.path.join(data_dir, "train_winequality-red.csv"))
    train_loader_wine = torch.utils.data.DataLoader(wine_train, batch_size=cfg["batch_size"], num_workers=2)

    # dataiter = iter(train_loader_bos)
    # data, target = dataiter.next()

    #Prepare model
    print("Prepare model")
    from model.fc import FC
    # model = FC(bos_train.input_dim, cfg["pdrop"])
    model = FC(wine_train.input_dim, cfg["pdrop"])
    model.cuda()

    #Prepare training
    print("Prepare training")
    optimizer = optim.Adam(model.parameters(), lr=tu.lr_scheduler())

    #Define starting iteration/epochs.
    #Will use checkpoints in the future when running on clusters
    if cfg["ckpt"] is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=cfg["ckpt"])
    elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    else:
        starting_iteration, starting_epoch = 0, 0

    #Logging
    tb_logger = create_tb_logger(output_dir)

    #Training
    print("Start training")
    trainer = Trainer(model=model,
                      model_fn=model_fn,
                      optimizer=optimizer,
                      ckpt_dir=ckpt_dir,
                      grad_norm_clip=cfg["grad_norm_clip"],
                      tb_logger=tb_logger)

    # trainer.train(num_epochs=cfg["num_epochs"],
    #               train_loader=train_loader_bos,
    #               ckpt_save_interval=cfg["ckpt_save_interval"],
    #               starting_iteration=starting_iteration,
    #               starting_epoch=starting_epoch)
    #
    trainer.train(num_epochs=cfg["num_epochs"],
                  train_loader=train_loader_wine,
                  ckpt_save_interval=cfg["ckpt_save_interval"],
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    #Testing
    print("Start testing")
    # Currently the test dataset is the same as training set. TODO: K-Flod cross validation
    # bos_test = BostonDataset(os.path.join(data_dir, "boston_test.csv"))
    # test_loader = torch.utils.data.DataLoader(bos_test, batch_size=cfg["batch_size"], num_workers=2)
    wine_test = WineDataset(os.path.join(data_dir, "test_winequality-red.csv"), testing=True)
    test_loader = torch.utils.data.DataLoader(wine_test, batch_size=cfg["batch_size"], num_workers=2)

    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(trainer._epoch + 1)))
    print("loading checkpoint ckpt_e{}".format(trainer._epoch + 1))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])

    test_loss = 0.0
    model.train()

    eval(model, test_loader=test_loader, cfg=cfg, tb_logger=tb_logger)

    #Finalizing
    print("Finished")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future