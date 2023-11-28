from setuptools import setup, find_packages

with open("acegen/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="acegen",
    version="1.0",
    license='MIT',
    author="Albert Bou",
    author_email="albertbou92@gmail.com",
    description="A torchrl RL framework for de novo drug design",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
    package_data={"priors": [
        "chembl_actor.prior",
        "chembl_actor_critic.prior",
        "chembl_critic.prior",
        "chembl_vocabulary.txt",
        "reinvent_vocabulary.txt",
        "zinc_actor_critic.prior",
        "zinc_vocabulary.prior",
    ]},
    install_requires=[
        # List your dependencies here
    ],
    python_requires=">=3.8",
)