#! /usr/bin/python3
import datetime
import logging
import os
import random
from glob import glob
from importlib import resources
from pathlib import Path

import hydra
import numpy as np
import torch

from acegen.data import load_dataset, SMILESDataset
from acegen.models import models, register_model
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary, tokenizer_options
from rdkit import Chem
from tensordict.utils import remove_duplicates
from tokenizer import Tokenizer
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.modules.utils import get_primers_from_module
from torchrl.record.loggers import get_logger
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    filename="pretraining.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# hydra outputs saved in /tmp
os.chdir("/tmp")


# Wraps the actor into a module for DDP
class Model(torch.nn.Module):
    def __init__(self, actor, device):
        super().__init__()
        self.actor = actor
        self.device = device
        self.actor.to(device)

    def forward(self, batch):

        batch = batch.to(self.device)
        target = batch.get("action")

        # Forward pass
        dist = self.actor.get_dist(batch)

        # Loss
        loss_actor = (-dist.log_prob(target) * batch["mask"]).sum(-1).mean()

        return loss_actor


def print_master(msg):
    barrier()
    if int(os.environ["RANK"]) == 0:
        print(msg)
        logging.info(msg)
    barrier()


@hydra.main(
    config_path=".",
    config_name="config",
    version_base="1.2",
)
def main(cfg: "DictConfig"):

    # Initialize processes
    if "WORLD_SIZE" not in os.environ:
        raise RuntimeError("the script has to be started with torchrun.")
    init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
    master = int(os.environ["RANK"]) == 0
    local_rank = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Set the default device
    # NOTE: this shouldn't be needed, but something is screwed down the stack
    torch.cuda.set_device(device)

    os.chdir(os.path.dirname(__file__))
    if master:
        seed = cfg.seed
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        os.makedirs(cfg.model_log_dir, exist_ok=True)

    logging.info("\nConstructing vocabulary...")
    if master:
        vocabulary = SMILESVocabulary.create_from_smiles(
            load_dataset(cfg.train_dataset_path),
            tokenizer=tokenizer_options[cfg.tokenizer](),
        )
        save_path = Path(cfg.model_log_dir) / "vocabulary.ckpt"
        torch.save(vocabulary.state_dict(), save_path)
    barrier()

    # Load vocabulary from a file
    vocabulary = SMILESVocabulary()
    vocabulary.load_state_dict(torch.load(save_path))
    vocabulary.tokenizer = Tokenizer()

    logging.info("\nPreparing dataset and dataloader...")
    if master:
        if cfg.recompute_dataset:
            logging.info("\nRemoving any existing previous dataset file...")
            for file in glob(f"{cfg.dataset_log_dir}/*.mmap"):
                os.remove(file)
        dataset = SMILESDataset(
            cache_path=cfg.dataset_log_dir,
            dataset_path=cfg.train_dataset_path,
            vocabulary=vocabulary,
            randomize_smiles=cfg.randomize_smiles,
        )
    barrier()

    # Create datasets in other processes
    dataset = SMILESDataset(
        cache_path=cfg.dataset_log_dir,
        dataset_path=cfg.train_dataset_path,
        vocabulary=vocabulary,
        randomize_smiles=cfg.randomize_smiles,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=DistributedSampler(dataset),
        drop_last=True,
        shuffle=False,  # Needs to be False with DistributedSampler
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    logging.info("\nCreating model...")
    # If custom model, register it
    if cfg.model not in models and cfg.get("custom_model_factory", None) is not None:
        register_model(cfg.model, cfg.model_factory)
    # Check if model is available
    if cfg.model not in models:
        raise ValueError(
            f"Model {cfg.model} not found. For custom models, define a model factory as explain in the tutorials."
        )
    # Get model
    create_model, _, _, _, _, _ = models[cfg.model]

    actor_training, actor_inference = create_model(vocabulary_size=len(vocabulary))
    actor_training = DistributedDataParallel(Model(actor_training, device))
    actor_inference.to(device)

    logging.info("\nCreating test environment...")
    test_env = SMILESEnv(
        start_token=vocabulary.start_token_index,
        end_token=vocabulary.end_token_index,
        length_vocabulary=len(vocabulary),
        batch_size=100,
        device=device,
    )
    test_env = TransformedEnv(test_env)
    test_env.append_transform(InitTracker())
    test_env.append_transform(get_primers_from_module(actor_inference))

    logging.info("\nCreating test scoring function...")

    def valid_smiles(smiles_list):
        result_tensor = torch.zeros(len(smiles_list))
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                result_tensor[i] = 1.0
        return result_tensor

    logging.info("\nCreating optimizer...")
    actor_optimizer = torch.optim.Adam(actor_training.parameters(), lr=cfg.lr)
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler)(
        actor_optimizer, **cfg.lr_scheduler_kwargs
    )

    logger = None
    if cfg.logger_backend:
        logging.info("\nCreating logger...")
        logger = get_logger(
            cfg.logger_backend,
            logger_name=Path.cwd(),
            experiment_name=cfg.agent_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.experiment_name,
                "group": cfg.agent_name,
            },
        )

    # Calculate number of parameters
    num_params = sum(param.numel() for param in actor_training.parameters())
    logging.info(f"Number of policy parameters {num_params:,}")

    logging.info("\nStarting pretraining...")
    actor_losses = torch.zeros(len(dataloader))
    for epoch in range(1, cfg.epochs):

        actor_losses.zero_()

        with tqdm(enumerate(dataloader), total=len(dataloader)) as tepoch:

            tepoch.set_description(f"Epoch {epoch}")

            for step, batch_td in tepoch:

                # Forward pass
                batch_td = batch_td.to(device)
                loss_actor = actor_training(batch_td)

                # Backward pass
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()
                actor_losses[step] = loss_actor.item()

            # Generate test smiles
            smiles = generate_complete_smiles(test_env, actor_inference, max_length=100)
            num_valid_smiles = valid_smiles(
                [vocabulary.decode(smi.cpu().numpy()) for smi in smiles.get("action")]
            ).sum()
            unique_smiles = remove_duplicates(smiles, key="action")

            # Log
            if logger and master:
                logger.log_scalar("loss_actor", actor_losses.mean(), step=epoch)
                logger.log_scalar("num_test_valid_smiles", num_valid_smiles, step=epoch)
                logger.log_scalar(
                    "num_test_unique_smiles", len(unique_smiles), step=epoch
                )
                logger.log_scalar("lr", lr_scheduler.get_lr()[0], step=epoch)

            # Decay learning rate
            lr_scheduler.step()

        save_path = Path(cfg.model_log_dir) / f"pretrained_actor_epoch_{epoch}.pt"
        torch.save(actor_training.state_dict(), save_path)

    print_master("Finished!")
    destroy_process_group()


if __name__ == "__main__":
    main()
