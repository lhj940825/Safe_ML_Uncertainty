'''
 * User: Hojun Lim
 * Date: 2020-07-05
'''

from utils.log_utils import create_tb_logger
from utils.eval_utils import *

if __name__ == '__main__':


    #TODO: Simplifiy and automate the process
    # TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = [40, 150]
    cfg["eval_epoch"] = [40, 150] # epoch of ckpt files that will be loaded for evaluation
    cfg["ckpt_save_interval"] = 20
    cfg["batch_size"] = 100
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 50 # number of samples to draw for MC Dropout
    cfg["pdrop"] = 0.1

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
    #output_dirs["year"] =  []

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
    test_datasets = {}
    test_loaders = {}

    for key, fname in data_files.items():
        train_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[0]), testing=True)
        train_loaders[key] = torch.utils.data.DataLoader(train_datasets[key], batch_size=cfg["batch_size"], num_workers=0, collate_fn = train_datasets[key].collate_batch)

        test_datasets[key] = UCIDataset(os.path.join(data_dirs[key], fname[2]), testing=True)
        test_loaders[key] = torch.utils.data.DataLoader(test_datasets[key],
                                                         batch_size=cfg["batch_size"],
                                                         num_workers=0,
                                                         collate_fn=test_datasets[key].collate_batch)

     # Prepare model
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
        mc_models[key].cuda()

    # Logging
    tb_loggers = {}
    for key, val in output_dirs.items():
        tb_loggers[key] = create_tb_logger(val)

    #Testing
    print("Start testing")
    # Currently the test dataset is the same as training set. TODO: K-Flod cross validation

    # loading trained models

    for ckpt_epoch in cfg["eval_epoch"]:
        cur_ckpts = {}
        for key, ckpt_dir in pu_ckpt_dirs.items():
            # cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(trainers[key]._epoch + 1)))
            # print("loading checkpoint ckpt_e{}".format(trainers[key]._epoch + 1))
            cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(ckpt_epoch)))
            print("loading checkpoint ckpt_e{}".format(ckpt_epoch))
            pu_models[key].load_state_dict(torch.load(cur_ckpts[key])["model_state"], strict=False)

            # Unlike MCDropout, we set the models evaluate mode here.
            pu_models[key].eval()

        for key, ckpt_dir in mc_ckpt_dirs.items():
            # cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(trainers[key]._epoch + 1)))
            # print("loading checkpoint ckpt_e{}".format(trainers[key]._epoch + 1))
            cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(ckpt_epoch)))
            print("loading checkpoint ckpt_e{}".format(ckpt_epoch))
            mc_models[key].load_state_dict(torch.load(cur_ckpts[key])["model_state"], strict=False)

            mc_models[key].train()



        pu_results = {}
        mc_results = {}
        for key in pu_models.keys():
            print("==================================Evaluating {}==========================================".format(key))
            #pu_models[key]
            #mc_models[key]
            pu_result, mc_result = ood_eval(pu_models[key], mc_models[key], test_loader=test_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='test-'+key, cur_epoch=ckpt_epoch)
            pu_results[key] = pu_result
            mc_results[key] = mc_result

            #TODO below function is to generate figures for training dataset as requested by Joachim
            #pu_eval_with_training_dataset(model, train_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='train-'+key)
            print("Finished\n")


        dataset_list = []
        pu_NLL_list = []
        pu_NLL_without_v_Noise_list = []
        pu_RMSE_list = []

        mc_NLL_list = []
        mc_NLL_without_v_Noise_list = []
        mc_RMSE_list = []


        for key, val in pu_results.items():
            dataset_list.append(key)
            pu_NLL_list.append(val[0][0])
            pu_RMSE_list.append(val[0][1])
            pu_NLL_without_v_Noise_list.append(val[3][0])

        for key, val in mc_results.items():
            mc_NLL_list.append(val[0][0])
            mc_RMSE_list.append(val[0][1])
            mc_NLL_without_v_Noise_list.append(val[3][0])

        err_df = pd.DataFrame(index=range(len(dataset_list)), columns=["Datasets", "pu_RMSE", "pu_NLL", "pu_NLL_no_v_noise", 'mc_RMSE', 'mc_NLL', 'mc_NLL_no_v_noise'])
        # a = pd.DataFrame(dataset_list)
        err_df["Datasets"] = pd.DataFrame(dataset_list)
        err_df["pu_RMSE"] = pd.DataFrame(pu_RMSE_list)
        err_df["pu_NLL"] = pd.DataFrame(pu_NLL_list)
        err_df["pu_NLL_no_v_noise"] = pd.DataFrame(pu_NLL_without_v_Noise_list)
        err_df["mc_RMSE"] = pd.DataFrame(mc_RMSE_list)
        err_df["mc_NLL"] = pd.DataFrame(mc_NLL_list)
        err_df["mc_NLL_no_v_noise"] = pd.DataFrame(mc_NLL_without_v_Noise_list)


        err_sum_dir = "./output_ood/err_summary"
        os.makedirs(err_sum_dir, exist_ok=True)
        err_df.to_csv(os.path.join(err_sum_dir, "ood_err_summary_epoch_"+str(ckpt_epoch)+".csv"), float_format='%.5f')


    #Finalizing
    print("Analysis finished\n")
