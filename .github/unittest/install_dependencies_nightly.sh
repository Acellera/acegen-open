python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm

# Not using nightly torch
python -m pip install torch torchvision
# python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall

# Not testing these dependencies for now
# python -m pip MolScore promptsmiles

cd ../acegen-open
pip install -e .
pip uninstall --yes torchrl
pip uninstall --yes tensordict

cd ..
python -m pip install git+https://github.com/pytorch-labs/tensordict.git
git clone https://github.com/pytorch/rl.git
cd rl
python setup.py develop
