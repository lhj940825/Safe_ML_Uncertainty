This project is developed for Lab course: Safe ML Uncertainty offered by Fraunhofer IAIS.

Developer: Hojun Lim. Zhengjiang Hu

This project can be run to conduct experiments of methods of uncertainty estimation: MC-dropout, Deep Ensemble and Parametric Uncertainty

This project mainly contains the following runnable scripts:

    safe_ml_lab_uncertainty/
        1. train_mc.py - training script for MC-dropout
        2. train_de.py - training script for Deep Ensemble
        3. train_pu.py - training script for Parametric Uncertainty
        4. train_ood.py - training script for OOD data study
        5. eval_mc.py - evaluation script for MC
        6. eval_de.py - evaluation script for DE
        7. eval_pu.py - evaluation script for PU
        8. eval_ood.py - evaluation script for OOD

        utils/
            9. dataset.py - script for generating datasets used for the experiments
    
To run specific experiment, simply run the corresponding training script first, then evaluation script.
Example: train_mc.py -> eval_mc.py for experiment of MC-dropout

Note that it is enough to run a training script only once as long as it remains unchanged. 
The corresponding evaluation script cannot be run if the training script has not been run for at least once.

The experiment results will be saved in the following root output directories:

    safe_ml_lab_uncertainty/
        1. output_mc - results of MC
        2. output_de - results of DE
        3. output_pu - results of PU
        4. output_ood - results of OOD

Each root output directory contains results of all datasets grouped in individual sub-directories named by the datasets' names.
    
(Optional) Data processing: The datasets have already been processed and saved in 'data', and 'data_ood' for normal datasets and OOD datasets. 
If one wants to generate the data himself, simply run "safe_ml_lab_uncertainty/utils/dataset.py". 
Please do not remove 'data' and 'data_ood' directories as they contain original datasets.
