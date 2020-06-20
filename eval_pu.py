'''
 * User: Hojun Lim
 * Date: 2020-06-06
'''

from utils.eval_utils import model_fn, model_fn_eval, eval, eval_with_training_dataset
import utils.train_utils as tu
from utils.dataset import *
from utils.log_utils import create_tb_logger
from utils.utils import *
from utils.eval_utils import *

if __name__ == '__main__':
    #TODO: Simplifiy and automate the process
    # TODO::put all kinds of cfgs and hyperparameter into a config file. e.g. yaml
    cfg = {}
    cfg["ckpt"] = None
    cfg["num_epochs"] = 150
    cfg["ckpt_save_interval"] = 20
    cfg["batch_size"] = 100
    cfg["grad_norm_clip"] = None
    cfg["num_networks"] = 10
    cfg["eval_epoch"] = 150

    #TODO: Simplifiy and automate the process
    #Create directory for storing results
    output_dirs = {}
    output_dirs["boston"] = []
    output_dirs["wine"] =  []
    #output_dirs["power_plant"] =  []
    #output_dirs["concrete"] =  []
    #output_dirs["energy"] = []
    #output_dirs["kin8nm"] = []
    #output_dirs["naval"] =  []
    #output_dirs["yacht"] =  []
    #output_dirs["protein"] =  []
    #output_dirs["year"] =  []

    for key, output_dir in output_dirs.items():
        output_dirs[key] = os.path.join('./output_pu', key, 'parametric_uncertainty')

    ckpt_dirs = {}
    for key, output_dir in output_dirs.items():
        ckpt_dirs[key] = os.path.join(output_dir, 'ckpts')
        os.makedirs(ckpt_dirs[key], exist_ok=True)

    data_dirs = {}
    for key, val in output_dirs.items():
        data_dirs[key] = os.path.join("./data", key)

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

    models = {}
    for key, dataset in train_datasets.items():
        if key in ["protein", "year"]:
            models[key] = pu_fc2(dataset.input_dim)
        else:
            models[key] = pu_fc(dataset.input_dim)
        models[key].cuda()


    # Logging
    tb_loggers = {}
    for key, val in output_dirs.items():
        tb_loggers[key] = create_tb_logger(val)

    #Testing
    print("Start testing")
    # Currently the test dataset is the same as training set. TODO: K-Flod cross validation

    # loading trained models
    cur_ckpts = {}
    for key, ckpt_dir in ckpt_dirs.items():
        # cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(trainers[key]._epoch + 1)))
        # print("loading checkpoint ckpt_e{}".format(trainers[key]._epoch + 1))
        cur_ckpts[key] = '{}.pth'.format(os.path.join(ckpt_dir, "ckpt_e{}".format(cfg["eval_epoch"])))
        print("loading checkpoint ckpt_e{}".format(cfg["eval_epoch"]))
        models[key].load_state_dict(torch.load(cur_ckpts[key])["model_state"], strict=False)

        # Unlike MCDropout, we set the models evaluate mode here.
        models[key].eval()


    # Summarize the result into table and save it
    results = {}
    for key, model in models.items():
        print("==================================Evaluating {}==========================================".format(key))
        result = pu_eval(model, test_loader=test_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='test-'+key)
        results[key] = result

        #TODO below function is to generate figures for training dataset as requested by Joachim
        pu_eval_with_training_dataset(model, train_loaders[key], cfg=cfg, output_dir=output_dirs[key], tb_logger=tb_loggers[key], title='train-'+key)
        print("Finished\n")



    # plot the gt-mean and std figures for every 10 epochs to see the training process of PU model
    ckpt_idxs = np.linspace(start=0, stop=cfg["num_epochs"], num=(cfg["num_epochs"]//10 +1), dtype=int) # ckpt_idxs = [0,10,20,...,150]
    for key, model in models.items():
        for idx in ckpt_idxs:
            cur_ckpt =  '{}.pth'.format(os.path.join(ckpt_dirs[key], "ckpt_e{}".format(idx)))
            model.load_state_dict(torch.load(cur_ckpt)["model_state"], strict=False)
            pu_eval_residualError_and_std_with_particular_epoch(model= model,train_loader=train_loaders[key], output_dir=output_dirs[key],title='train-'+key+'-epoch='+str(idx))



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
    # a = pd.DataFrame(dataset_list)
    err_df["Datasets"] = pd.DataFrame(dataset_list)
    err_df["RMSE"] = pd.DataFrame(RMSE_list)
    err_df["NLL"] = pd.DataFrame(NLL_list)
    err_df["NLL_no_v_noise"] = pd.DataFrame(NLL_without_v_Noise_list)


    err_sum_dir = "./output_pu/err_summary"
    os.makedirs(err_sum_dir, exist_ok=True)
    err_df.to_csv(os.path.join(err_sum_dir, "pu_err_summary.csv"))

    plot_NLL_cap_cnt(dataset_list, NLL_over_cap_cnt, cap, err_sum_dir)

    #Finalizing
    print("Analysis finished\n")
    #TODO: integrate logging, visualiztion, GPU data parallel etc in the future