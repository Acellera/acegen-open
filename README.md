# torchrl_chem
Language models for drug discovery using torchrl


## Running the working code

    python ppo.py

## To reproduce the error with ParallelEnv

For some reason using ParallelEnv causes the hidden states to be always zero.

    python ppo_reproduce_error.py