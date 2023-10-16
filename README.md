# TorchRL generative Chemistry
Language models + RL for drug discovery using TorchRL

## Conda environment

To create the conda / mamba environment, run

    conda/mamba env create -f environment.yml
    conda activate torchrl_chem

## Configuration file
    
The configuration file is `config.yml`. It contains the parameters for the training.

## Running the training script

To run the training script with kl divergence as a loss term, run

    python ppo_kl_loss.py

To run the training script with kl divergence as a reward term, run

    python ppo_kl_reward.py

## To see the results

    python analyse_logs.py --logs-path path/to/logs/dir