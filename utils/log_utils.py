from torch.utils.tensorboard import SummaryWriter
import os

def create_tb_logger(root_dir, tb_log_dir='tensorboard'):
    return SummaryWriter(log_dir=os.path.join(root_dir, tb_log_dir))
