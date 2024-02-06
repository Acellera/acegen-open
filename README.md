# TorchRL generative Chemistry
Generative chemistry models + RL for drug discovery using TorchRL

## Overview

## Installation

### Conda environment

To create the conda / mamba environment, run

    conda create -n acegen python=3.10 -y
    conda activate acegen
    pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
    
### Install Tensordict

To install Tensordict, run

    git clone https://github.com/pytorch/tensordict.git
    cd tensordict
    python setup.py install

### Install TorchRL

To install TorchRL, run

    git clone https://github.com/pytorch/rl.git
    cd rl
    python setup.py install

### Install MolScore

To install MolScore, run

    pip3 install rdkit func_timeout dask distributed pystow zenodo_client matplotlib scipy pandas joblib seaborn molbloom Levenshtein
    git clone https://github.com/MorganCThomas/MolScore.git
    cd molscore
    python setup.py install

### Install AceGen

To install AceGen, run

    pip3 install tqdm wandb hydra-core
    cd acegen-open
    python setup.py install


## Running training scripts

To run the training scripts, run

    python examples/a2c/a2c.py
    python examples/ppo/ppo.py
    python examples/reinvent/reinvent.py
    python examples/ahc/ahc.py
