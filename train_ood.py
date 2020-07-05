'''
 * User: Hojun Lim
 * Date: 2020-07-05


 Train Parametric Uncertainty models and MC Dropout models with small 50% labels(later evaluate with large 50% labels: indicating the ood dataset)
'''

from torch import optim
from utils.train_utils import Trainer, load_checkpoint
from utils.eval_utils import model_fn_for_pu, model_fn_eval, eval, eval_with_training_dataset, model_fn
from utils.dataset import *
from utils.log_utils import create_tb_logger
from utils.utils import *


if __name__ == '__main__':
    # TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 40
    cfg["ckpt_save_interval"] = 10
    cfg["batch_size"] = 100
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 10
    cfg["pdrop"] = 0.1
    learning_rate = {'PU' : 0.001, 'MC':0.1}

    #TODO: Simplifiy and automate the process
    #Create directory for storing results
    output_dirs = {}

    output_dirs["boston"] = []
    output_dirs["wine"] =  []
    output_dirs["power_plant"] =  []
    output_dirs["concrete"] =  []
    output_dirs["energy"] = []
    output_dirs["kin8nm"] = []
    output_dirs["naval"] =  []
    output_dirs["yacht"] =  []
    output_dirs["protein"] =  []
    # output_dirs["year"] =  []

    for key, output_dir in output_dirs.items():
        output_dirs[key] = os.path.join('./output_ood', key)
        os.makedirs(output_dirs[key], exist_ok=True)

    pu_ckpt_dirs = {}
    mc_ckpt_dirs = {}
    for key, output_dir in output_dirs.items():
        pu_ckpt_dirs[key] = os.path.join(output_dir, 'ckpts', 'pu')
        mc_ckpt_dirs[key] = os.path.join(output_dir, 'ckpts', 'mc')

        os.makedirs(pu_ckpt_dirs[key], exist_ok=True)
        os.makedirs(mc_ckpt_dirs[key], exist_ok=True)

    data_dirs = {}
    for key, val in output_dirs.items():
        data_dirs[key] = os.path.join("./data_ood", key)

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


        eval_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[0]), testing=True)
        eval_loaders[key] = torch.utils.data.DataLoader(eval_datasets[key],
                                                        batch_size=cfg["batch_size"],
                                                        num_workers=0,
                                                        collate_fn=eval_datasets[key].collate_batch)

    #Prepare model
    print("Prepare model")
    from model.pu_fc import pu_fc, pu_fc2
    from model.fc import FC, FC2

    mc_models = {}
    pu_models = {}
    for key, dataset in train_datasets.items():
        if key in ["protein", "year"]:
            pu_models[key] = pu_fc2(dataset.input_dim)
            mc_models[key] = FC2(dataset.input_dim, cfg["pdrop"])
        else:
            pu_models[key] = pu_fc(dataset.input_dim)
            mc_models[key] = FC(dataset.input_dim, cfg["pdrop"])

        pu_models[key].cuda()
        pu_models[key].train()

        mc_models[key].cuda()
        mc_models[key].train()


    #Prepare training
    print("Prepare training")
    pu_optimizers = {}
    mc_optimizers = {}
    for key, model in pu_models.items():
        pu_optimizers[key] = optim.Adam(model.parameters(), lr=learning_rate['PU'])

    for key, model in mc_models.items():
        mc_optimizers[key] = optim.Adam(model.parameters(), lr=learning_rate['MC'])

    starting_iteration, starting_epoch = 0, 0

    #Logging
    tb_loggers = {}
    for key, val in output_dirs.items():
        tb_loggers[key] = create_tb_logger(val)


    #Training
    print("Start training")

    pu_trainers = {}

    for key, model in pu_models.items():
        print(key, model)
        print("*******************************Training pu_models: {}*******************************\n".format(key))
        pu_trainers[key] = Trainer(model=model,
                                   model_fn=model_fn_for_pu,
                                   model_fn_eval=model_fn_eval,
                                   optimizer=pu_optimizers[key],
                                   ckpt_dir=pu_ckpt_dirs[key],
                                   output_dir=output_dirs[key],
                                   title='pu_train_{}'.format(key),
                                   grad_norm_clip=cfg["grad_norm_clip"],
                                   tb_logger=tb_loggers[key])


        pu_trainers[key].train(num_epochs=cfg["num_epochs"],
                               train_loader=train_loaders[key],
                               eval_loader=eval_loaders[key],
                               # eval_loader=None,
                               ckpt_save_interval=cfg["ckpt_save_interval"],
                               starting_iteration=starting_iteration,
                               starting_epoch=starting_epoch)


        # draw_loss_trend_figure(key, len(trainers[key].train_loss), trainers[key].train_loss, trainers[key].eval_loss, output_dirs[key])
        draw_loss_trend_figure("PU "+ key, len(pu_trainers[key].train_loss), pu_trainers[key].train_loss, output_dir=output_dirs[key])
        print("*******************************Finished training {}*******************************\n".format(key))


    mc_trainers = {}
    for key, model in mc_models.items():
        print(key, model)
        print("*******************************Training Mc models: {}*******************************\n".format(key))
        mc_trainers[key] = Trainer(model=model,
                                   model_fn= model_fn,
                                   model_fn_eval=model_fn_eval,
                                   optimizer=mc_optimizers[key],
                                   ckpt_dir=mc_ckpt_dirs[key],
                                   output_dir=output_dirs[key],
                                   title='mc_train_{}'.format(key),
                                   grad_norm_clip=cfg["grad_norm_clip"],
                                   tb_logger=tb_loggers[key])


        mc_trainers[key].train(num_epochs=cfg["num_epochs"],
                               train_loader=train_loaders[key],
                               # eval_loader=eval_loaders[key],
                               # eval_loader=None,
                               ckpt_save_interval=cfg["ckpt_save_interval"],
                               starting_iteration=starting_iteration,
                               starting_epoch=starting_epoch)


        # draw_loss_trend_figure(key, len(trainers[key].train_loss), trainers[key].train_loss, trainers[key].eval_loss, output_dirs[key])
        draw_loss_trend_figure("MC "+ key, len(mc_trainers[key].train_loss), mc_trainers[key].train_loss, output_dir=output_dirs[key])
        print("*******************************Finished training {}*******************************\n".format(key))

    #Finalizing
    print("Training finished\n")