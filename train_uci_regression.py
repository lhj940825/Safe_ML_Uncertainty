import torch
import torchvision
from torch import optim
import os
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn
import utils.train_utils as tu
import utils.dataset
import utils as u
import cv2 as cv
import numpy as np

if __name__ == "__main__":

    #Create directory for storing results
    output_dir = os.path.join("./", "output", "uci_regression")
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    #some cfgs, some cfg will be used in the future
    #TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    ckpt = None
    num_epochs = 20
    ckpt_save_interval = 10
    batch_size = 5

    #Prepare data Boston Housing coverted to pytorch dataset
    dataset_bos, input_dim = u.dataset.boston_dataset()
    train_loader_bos = torch.utils.data.DataLoader(dataset_bos, batch_size=batch_size, num_workers=2)

    # dataiter = iter(train_loader_bos)
    # data, target = dataiter.next()

    #Prepare model
    print("Prepare model")
    from model.fc import FC
    model = FC(input_dim)
    model.cuda()

    #Prepare training
    print("Prepare training")
    optimizer = optim.SGD(model.parameters(), lr=tu.lr_scheduler(), momentum=0.9)

    #Define starting iteration/epochs.
    #Will use checkpoints in the future when running on clusters
    if ckpt is not None:
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=ckpt)
    elif os.path.isfile(os.path.join(ckpt_dir, "sigterm_ckpt.pth")):
        starting_iteration, starting_epoch = load_checkpoint(model=model, optimizer=optimizer, filename=os.path.join(ckpt_dir, "sigterm_ckpt.pth"))
    else:
        starting_iteration, starting_epoch = 0, 0

    #Training
    print("Start training")
    trainer = Trainer(model=model,
                      model_fn=model_fn,
                      optimizer=optimizer,
                      ckpt_dir=ckpt_dir)

    trainer.train(num_epochs=num_epochs,
                  train_loader=train_loader_bos,
                  ckpt_save_interval=ckpt_save_interval,
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    #Testing
    print("Start testing")
    # Currently the test dataset is the same as training set. TODO: K-Flod cross validation
    test_loader_bos = torch.utils.data.DataLoader(dataset_bos, batch_size=batch_size, num_workers=2)

    cur_ckpt = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e%d" % (trainer._epoch + 1)))
    model.load_state_dict(torch.load(cur_ckpt)["model_state"])

    with torch.no_grad():
        from sklearn.datasets import load_boston
        boston = load_boston()
        data = torch.tensor(boston.data, dtype=torch.float).cuda()
        target = torch.tensor(boston.target, dtype=torch.float).view(-1, 1).cuda()
        loss = torch.nn.MSELoss()
        print(loss(model(data), target).item())

        print(model(data[0]).data)
        print(target[0])
        # for data in test_loader_bos:
        #     data, target = data
        #     data = data.cuda()
        #     target = target.cuda()
        #     outputs = model(data)
        #     _, predicted = torch.max(outputs.data.cpu(), 1)
        #     total += labels.size(0)
        #     correct += (predicted == labels.cpu()).sum().item()

    #Finalizing
    print("Finished")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future