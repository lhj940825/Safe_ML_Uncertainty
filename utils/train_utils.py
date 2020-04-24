import torch
import os

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
    return 0.001

class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir):
        self.model = model
        self.model_fn = model_fn
        self.optimizer = optimizer
        self.ckpt_dir = ckpt_dir

        self._epoch = 0
        self._it = 0

    def train(self, num_epochs, train_loader, ckpt_save_interval=3, starting_epoch=0, starting_iteration=0):
        for self._epoch in range(num_epochs):
            running_loss = 0.0
            #Train for one epoch
            for cur_it, batch in enumerate(train_loader):
                loss = self._train_it(batch)
                running_loss += loss
                self._it += 1

            #save trained model
            trained_epoch = self._epoch + 1
            print("Current Epoch: %d" % trained_epoch)
            # a = len(train_loader)
            print("Epoch loss: ", running_loss / len(train_loader))
            if trained_epoch % ckpt_save_interval == 0:
                print("Saving checkpoint")
                ckpt_name = os.path.join(self.ckpt_dir, "ckpt_e%d" % trained_epoch)
                save_checkpoint(checkpoint_state(self.model, self.optimizer, trained_epoch, self._it), filename=ckpt_name)

    def _train_it(self, batch):
        self.model.train()  #Set the model to training mode
        self.optimizer.zero_grad()  #Clear the gradients before training

        loss = self.model_fn(self.model, batch)

        loss.backward()
        self.optimizer.step()

        return loss.item()