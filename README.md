# TorchRL generative Chemistry
Language models + RL for drug discovery using TorchRL

## Conda environment

To create the conda / mamba environment, run

    conda/mamba env create -f environment.yml
    conda activate torchrl_chem

## Install AceGen

To install AceGen, run

    cd acegen
    python setup.py install

## Run examples

### Configuration file
    
The configuration file is `ppo_config.yml`. It contains the parameters for the training.

### Running the training script

To run the training script with kl divergence as a loss term, run

    python ppo.py
