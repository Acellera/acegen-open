import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from acegen.data import load_dataset, smiles_to_tensordict, SMILESDataset
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from rdkit import Chem
from tensordict.utils import remove_duplicates
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.record.loggers import get_logger
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    filename="pretraining.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = f"cuda:0" if torch.cuda.device_count() > 1 else "cpu"
    os.makedirs(cfg.model_log_dir, exist_ok=True)

    logging.info("\nConstructing vocabulary...")
    vocabulary = SMILESVocabulary.create_from_smiles(
        load_dataset(cfg.train_dataset_path),
        tokenizer=Tokenizer(),
    )
    save_path = Path(cfg.model_log_dir) / "vocabulary.ckpt"
    torch.save(vocabulary.state_dict(), save_path)

    logging.info("\nPreparing dataset and dataloader...")
    dataset = SMILESDataset(
        cache_path=cfg.dataset_log_dir,
        dataset_path=cfg.train_dataset_path,
        vocabulary=vocabulary,
        randomize_smiles=cfg.randomize_smiles,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    logging.info("\nCreating model...")

    if cfg.model == "lstm":
        from acegen.models import create_lstm_actor

        create_model = create_lstm_actor
    elif cfg.model == "gru":
        from acegen.models import create_gru_actor

        create_model = create_gru_actor
    elif cfg.model == "gpt2":
        from acegen.models import create_gpt2_actor

        create_model = create_gpt2_actor
    else:
        raise ValueError(f"Unknown model type {cfg.model}")

    actor_training, actor_inference = create_model(vocabulary_size=len(vocabulary))
    actor_training.to(device)
    actor_inference.to(device)

    logging.info("\nCreating test environment...")
    # Create a transform to populate initial tensordict with rnn recurrent states equal to 0.0
    primers = actor_training.rnn_spec.expand(cfg.num_test_smiles)
    rhs_primer = TensorDictPrimer(primers)
    test_env = SMILESEnv(
        start_token=vocabulary.start_token_index,
        end_token=vocabulary.end_token_index,
        length_vocabulary=len(vocabulary),
        batch_size=100,
        device=device,
    )
    test_env = TransformedEnv(test_env)
    test_env.append_transform(InitTracker())
    test_env.append_transform(rhs_primer)

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
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        actor_optimizer, step_size=1, gamma=cfg.lr_decay_per_epoch
    )

    logger = None
    if cfg.logger_backend:
        logging.info("\nCreating logger...")
        logger = get_logger(
            cfg.logger_backend,
            logger_name="pretrain",
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

            for step, batch in tepoch:

                batch = batch.to(device)
                batch_td = smiles_to_tensordict(
                    batch, replace_mask_value=0, device=device
                )
                batch.set("sequence", batch.get("observation"))
                target = batch_td.get("action")
                batch_td.set("is_init", torch.zeros_like(target).unsqueeze(-1).bool())

                # Forward pass
                dist = actor_training.get_dist(batch_td)

                # Loss
                loss_actor = (-dist.log_prob(target) * batch_td["mask"]).sum(-1).mean()

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
            if logger:
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


if __name__ == "__main__":
    main()
