# TorchRL generative Chemistry
Language models + RL for drug discovery using TorchRL

## Conda environment

To create the conda / mamba environment, run

    conda/mamba env create -f environment.yml
    conda activate torchrl_chem

## Install AceGen

To install AceGen, run

    cd acegen-open
    python setup.py install

## Install MolScore

To install MolScore, run
    
    git clone https://github.com/MorganCThomas/MolScore.git
    cd molscore
    python setup.py install

## Run examples

### Configuration file
    
The configuration file is `ppo_config.yml`. It contains the parameters for the training.

### Running the training script

To run the training script with kl divergence as a loss term, run

    python ppo.py
