# AceGen: A TorchRL-based toolkit for reinforcement learning in generative chemistry

## Overview

In recent years, reinforcement learning (RL) has been increasingly used in drug design to propose molecules with specific properties under defined constraints. However, RL problems are inherently complex, featuring independent and interchangeable components with diverse method signatures and data requirements.

AceGen applies TorchRL - a modern general decision-making library that provides well-integrated reusable components - to make a robust toolkit tailored for generative drug design.

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
    git checkout develop
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

To modify training parameters, edit the corresponding YAML file in each example's directory.

# Available models

We provide a variety of example priors that can be selected in the configuration file. These include:

- A Gated Recurrent Unit (GRU) model
    - pre-training dataset: [ChEMBL](https://www.ebi.ac.uk/chembl/)
    - number of parameters: 4,363,045


- A Long Short-Term Memory (LSTM) model
    - pre-training dataset: [ZINC250k](https://github.com/wenhao-gao/mol_opt/blob/main/data/zinc.txt.gz)
    - number of parameters: 5,807,909
 

- A GPT-2 model
    - pre-training dataset: [REAL 350/3 lead-like, 613.86M cpds, CXSMILES](https://enamine.net/compound-collections/real-compounds/real-database-subsets)
    - number of parameters: XXXXXX
