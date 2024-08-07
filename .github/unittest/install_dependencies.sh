python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm

# Not using nightly torch
python -m pip install torch torchvision
# python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall

python -m pip torchrl
python -m pip install rdkit==2023.3.3
python -m pip MolScore promptsmiles

cd ../acegen-open
pip install -e .
