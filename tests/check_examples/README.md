# Check examples

This folder contains a `run_scripts.sh` file that executes all
the training scripts using `sbatch` with the default configuration and logs them
into a common WandB project.

## Usage

To display the script usage, you can use the `--help` option:

```bash
./run_scripts.sh --help
```

## Setup

The following setup should allow you to run the scripts:

```bash
conda create -n acegen-examples-check python=3.10 -y 
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install tqdm wandb hydra-core

cd /path/to/tensordict
python setup.py develop
cd /path/to/torchrl
python setup.py develop
cd /path/to/acegen-open
python setup.py develop
cd /path/to/MolScore
python setup.py develop
```
