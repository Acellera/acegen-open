#! /usr/bin/python3
import logging
import os
import random
from glob import glob
from importlib import resources
from pathlib import Path

import hydra
import numpy as np
import torch

from acegen.data import chem_utils, load_dataset, MolBloomDataset, SMILESDataset
from acegen.models import models, register_model
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary, tokenizer_options
from tensordict.utils import remove_duplicates
from torch.utils.data import DataLoader
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.modules.utils import get_primers_from_module
from torchrl.record.loggers import get_logger
from tqdm import tqdm

try:
    import wandb

    _has_wandb = True
except:
    _has_wandb = False

# hydra outputs saved in /tmp
os.chdir("/tmp")


@hydra.main(
    config_path=".",
    config_name="config",
    version_base="1.2",
)
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = f"cuda:0" if torch.cuda.device_count() >= 1 else "cpu"
    os.chdir(os.path.dirname(__file__))
    os.makedirs(cfg.model_log_dir, exist_ok=True)

    logging.info("\nConstructing vocabulary...")
    vocabulary = SMILESVocabulary.create_from_smiles(
        load_dataset(cfg.train_dataset_path),
        tokenizer=tokenizer_options[cfg.tokenizer](),
        special_tokens=cfg.get("special_tokens", []),
    )
    save_path = Path(cfg.model_log_dir) / "vocabulary.ckpt"
    torch.save(vocabulary.state_dict(), save_path)

    if cfg.recompute_dataset:
        logging.info("\nRemoving any existing previous dataset file...")
        for file in glob(f"{cfg.dataset_log_dir}/*.mmap"):
            os.remove(file)

    logging.info("\nPreparing dataset and dataloader...")
    dataset = SMILESDataset(
        cache_path=cfg.dataset_log_dir,
        dataset_path=cfg.train_dataset_path,
        vocabulary=vocabulary,
        randomize_smiles=cfg.randomize_smiles,
    )
    molbloom_dataset = MolBloomDataset(dataset_path=cfg.train_dataset_path)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
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
    actor_training.to(device)
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
    if primers := get_primers_from_module(actor_inference):
        test_env.append_transform(primers)

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
            logger_name=cfg.model_log_dir,
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
                batch_td = batch_td.to(device)
                target = batch_td.get("action")

                # Forward pass
                dist = actor_training.get_dist(batch_td)

                # Loss
                loss_actor = (-dist.log_prob(target) * batch_td["mask"]).sum(-1).mean()

                # Backward pass
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()
                actor_losses[step] = loss_actor.item()

                if step % cfg.log_frequency == 0:
                    # Generate test smiles
                    smiles = generate_complete_smiles(
                        environment=test_env,
                        vocabulary=vocabulary,
                        policy_sample=actor_inference,
                        policy_evaluate=actor_training,
                        max_length=100,
                    )
                    smiles_log_prob = smiles["sample_log_prob"].sum(-1)
                    smiles_str = [
                        vocabulary.decode(smi.cpu().numpy())
                        for smi in smiles.get("action")
                    ]
                    mols = [chem_utils.get_mol(smi) for smi in smiles_str]
                    valid_smiles = chem_utils.fraction_valid(mols)
                    unique_smiles = len(remove_duplicates(smiles, key="action")) / len(
                        smiles
                    )
                    inside_smiles = np.mean(
                        [smi in molbloom_dataset for smi in smiles_str]
                    )
                    total_smiles = ((epoch - 1) * (len(dataset))) + (
                        step * cfg.batch_size
                    )

                    # Log
                    if logger:
                        logger.log_scalar("num_smiles", total_smiles, step=total_smiles)
                        logger.log_scalar(
                            "loss_actor", actor_losses[step], step=total_smiles
                        )
                        logger.log_scalar(
                            "loss_sample", -smiles_log_prob.mean(), step=total_smiles
                        )
                        logger.log_scalar(
                            "valid_smiles", valid_smiles, step=total_smiles
                        )
                        logger.log_scalar(
                            "unique_smiles", unique_smiles, step=total_smiles
                        )
                        logger.log_scalar(
                            "inside_smiles", inside_smiles, step=total_smiles
                        )
                        if _has_wandb and cfg.logger_backend == "wandb":
                            image = chem_utils.draw(
                                np.random.choice(mols, 10, replace=False)
                            )
                            logger.log_scalar(
                                "mols", wandb.Image(image), step=total_smiles
                            )
                        logger.log_scalar(
                            "lr", lr_scheduler.get_last_lr()[0], step=total_smiles
                        )

                # Decay learning rate
                lr_scheduler.step()

        save_path = Path(cfg.model_log_dir) / f"pretrained_actor_epoch_{epoch}.pt"
        torch.save(actor_training.state_dict(), save_path)


if __name__ == "__main__":
    main()
