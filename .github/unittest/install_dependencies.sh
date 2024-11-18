# Upgrade pip
python -m pip install --upgrade pip

# Install required dependencies
python -m pip install flake8 pytest pytest-cov hydra-core tqdm
python -m pip install torch torchvision
python -m pip install git+https://github.com/pytorch/rl.git@767a877a0a93d41a9b7935598da0ded4c984904f
python -m pip install git+https://github.com/pytorch/tensordict.git@3812ca62d0d5db7d337592d99934e32a3d2b4bfd

# Install additional dependencies
python -m pip install transformers promptsmiles MolScore
python -m pip install deepsmiles
python -m pip install selfies
python -m pip install smi2sdf
python -m pip install smi2svg
python -m pip install atomInSmiles
python -m pip install safe-mol
python -m pip install smizip
python -m pip install molbloom
python -m pip install wheel
python -m pip install causal-conv1d>=1.4.0 --no-build-isolation
python -m pip install mamba-ssm==1.2.2 --no-build-isolation

# Verify installations
python -c "import transformers; print(transformers.__version__)"
python -c "import promptsmiles"
# python -c "import mamba_ssm; print('mamba-ssm:', mamba_ssm.__version__)"  # Assuming mamba-ssm imports as mamba

# Install local package
cd ../acegen-open
pip install -e .
