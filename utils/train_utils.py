import torch
import os
from torch.nn.utils import clip_grad_norm_
import yaml
from utils.utils import draw_loss_trend_figure, plot_sequence_mean_var
import numpy as np
from utils.eval_utils import pu_eval_with_training_dataset
from utils.utils import draw_loss_trend_figure


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}

def save_checkpoint(state=None, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=None):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        print('Could not find %s' % filename)
        raise FileNotFoundError

    return it, epoch

def lr_scheduler():
    return 0.1

class Trainer(object):
    def __init__(self, model, model_fn, model_fn_eval, optimizer, ckpt_dir, grad_norm_clip=1.0, tb_logger=None, output_dir='./tmp', title=''):
        self.model = model
        self.model_fn = model_fn
        self.model_fn_eval = model_fn_eval
        self.optimizer = optimizer
        self.ckpt_dir = ckpt_dir
        self.grad_norm_clip = grad_norm_clip
        self.tb_logger = tb_logger
        self.output_dir = output_dir
        self.title = title

        self._epoch = 0
        self._it = 0
        self.train_loss = []
        self.eval_loss = []

    def train(self, num_epochs, train_loader, eval_loader=None, ckpt_save_interval=3, starting_epoch=0, starting_iteration=0):
        if eval_loader is not None:
            seq_mean, seq_var = [], []

        # save the initial model
        ckpt_name = os.path.join(self.ckpt_dir, "ckpt_e{}".format(0))
        save_checkpoint(checkpoint_state(self.model, self.optimizer, 0, self._it), filename=ckpt_name)

        for self._epoch in range(num_epochs):
            running_loss = 0.0
            all_pred = torch.tensor([])
            #Train for one epoch
            for cur_it, batch in enumerate(train_loader):
                loss = self._train_it(batch)

                # Log to tensorboard
                if self.tb_logger is not None:
                    self.tb_logger.add_scalar('Loss/train_loss', loss, self._it)
                running_loss += loss
                self._it += 1

            if self.tb_logger is not None:
                self.tb_logger.flush()

            #save trained model
            trained_epoch = self._epoch + 1
            print("Current Epoch: %d" % trained_epoch)
            self.train_loss.append(running_loss / len(train_loader))
            print("Epoch loss: ", self.train_loss[self._epoch])


            if trained_epoch % ckpt_save_interval == 0:
                print("Saving checkpoint")
                ckpt_name = os.path.join(self.ckpt_dir, "ckpt_e{}".format(trained_epoch))
                save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, self._it), filename=ckpt_name)

#            if self.epoch % 10 ==0:
#                pu_eval_with_training_dataset(model= self.model, train_loader=train_loader,cfg=None, output_dir=None, tb_logger=None, title= )

            # eval one epoch
            if eval_loader is not None:
                with torch.set_grad_enabled(False):
                    mean_list, var_list = self.model_fn_eval(self.model, eval_loader)
                    seq_mean.append(mean_list)
                    seq_var.append(var_list)

        if eval_loader is not None:
            seq_mean = np.array(seq_mean)
            seq_std = np.array(seq_var)
            plot_sequence_mean_var(seq_mean, seq_var, output_dir=os.path.join(self.output_dir, 'net_output_scat_sequence'), title=self.title)

    def _train_it(self, batch):
        self.model.train()  #Set the model to training mode
        self.optimizer.zero_grad()  #Clear the gradients before training

        loss = self.model_fn(self.model, batch)

        loss.backward()
        if self.grad_norm_clip:
            print([param.weight.grad for param in self.model.parameters()])
            clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        #print('before update',[param.data for param in self.model.parameters()])
        self.optimizer.step()
        #print('after update',[param.data for param in self.model.parameters()])
        return loss.item()






