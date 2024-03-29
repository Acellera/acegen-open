from setuptools import find_packages, setup

setup(
    packages=find_packages(),
    include_package_data=True,
    package_data={"acegen": [
        "priors/**",
        "scripts/**/config_denovo.yaml",
        "scripts/**/config_fragment.yaml",
        "scripts/**/config_scaffold.yaml",
        ]},
    scripts=[
        "scripts/a2c/a2c.py",
        "scripts/ahc/ahc.py",
        "scripts/ppo/ppo.py",
        "scripts/reinvent/reinvent.py",
        "scripts/pretrain/pretrain_single_node.py",
        "scripts/pretrain/pretrain_distributed.py",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=[
        "torch",
        "tensordict",
        "torchrl",
        "tqdm",
        "wandb",
        "hydra-core"
    ]
)
