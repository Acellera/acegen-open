# TorchRL generative Chemistry
Language models + RL for drug discovery using TorchRL

## Conda environment

To create the conda environment, run

    conda env create -f environment.yml

## Configuration file
    
The configuration file is `config.yml`. It contains the parameters for the training.

## Running the training script

    python ppo_kl_loss.py

or 

    python ppo_kl_reward.py

## To see the results

    python analyze.py