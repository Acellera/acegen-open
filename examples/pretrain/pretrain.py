from pathlib import Path

import hydra
import torch
from acegen.dataset import load_dataset, SMILESDataset
from acegen.models import create_gru_actor, create_gru_critic
from acegen.vocabulary import SMILESVocabulary
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.record.loggers import get_logger
from tqdm import tqdm


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    device = f"cuda:0" if torch.cuda.device_count() > 1 else "cpu"

    cfg.train_dataset_path = (
        Path(__file__).resolve().parent.parent.parent / "priors" / "smiles_test_set"
    )

    print("\nConstructing vocabulary...")
    vocabulary = SMILESVocabulary.create_from_smiles(
        load_dataset(cfg.train_dataset_path)
    )

    print("\nPreparing dataset and dataloader...")
    dataset = SMILESDataset(
        cache_path=cfg.dataset_log_dir,
        dataset_path=cfg.train_dataset_path,
        vocabulary=vocabulary,
        randomize_smiles=cfg.randomize_smiles,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        # sampler=Sampler(dataset),  # Sampler option is mutually exclusive with shuffle
        shuffle=True,  # Needs to be False with DistributedSampler
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    print("\nCreating model...")
    actor_training, actor_inference = create_gru_actor(len(vocabulary))
    # critic_training, critic_inference = create_gru_critic(len(vocabulary))
    actor_training.to(device)
    # critic_training.to(device)
    actor_inference.to(device)
    # critic_inference.to(device)

    print("\nCreating optimizer...")
    actor_optimizer = torch.optim.Adam(actor_training.parameters(), lr=cfg.lr)
    # critic_optimizer = torch.optim.Adam(critic_training.parameters(), lr=cfg.lr)

    logger = None
    if cfg.logger:
        print("\nCreating logger...")
        logger = get_logger(
            cfg.logger,
            logger_name="pretrain",
            experiment_name=cfg.agent_name,
            project_name=cfg.experiment_name,
        )

    # Calculate number of parameters
    num_params = sum(param.numel() for param in actor_training.parameters())
    print(f"Number of policy parameters {num_params}")

    print("\nStarting pretraining...")
    actor_losses = torch.zeros(len(dataloader))
    for epoch in range(1, cfg.epochs):

        actor_losses.zero_()

        with tqdm(enumerate(dataloader), total=len(dataloader)) as tepoch:

            tepoch.set_description(f"Epoch {epoch}")

            for step, batch in tepoch:

                batch = batch.to(device)
                mask = (batch != -1).float()  # Mask padding tokens

                num_layers = 3
                hidden_size = 512
                td_batch = TensorDict(
                    {
                        "observation": batch.long() * mask.long(),
                        "is_init": torch.zeros_like(batch).bool(),
                        "recurrent_state": torch.zeros(
                            *batch.shape[:2], num_layers, hidden_size
                        ),
                    },
                    batch_size=batch.shape[:2],
                )

                # Forward pass
                td_batch = actor_training(td_batch)
                # td_batch = critic_training(td_batch)

                # Loss
                loss_actor = (
                    (-td_batch.get("sample_log_prob").squeeze(-1) * mask).sum(0).mean()
                )
                # loss_critic = 0.0

                # Backward pass
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()
                # critic_optimizer.zero_grad()
                # loss_critic.backward()
                # critic_optimizer.step()

                actor_losses[step] = loss_actor.item()

            # Log
            if logger:
                logger.log_scalar("loss_actor", actor_losses.mean())
                # logger.log_scalar("loss_critic", loss_critic.item())

        save_path = Path(cfg.model_log_dir) / f"pretrained_actor_epoch_{epoch}.pt"
        torch.save(actor_training.state_dict(), save_path)


if __name__ == "__main__":
    main()
