from pathlib import Path

import hydra
import torch
import tqdm
import wandb
from acegen.dataset import load_dataset, SMILESDataset
from acegen.models import create_gru_actor, create_gru_critic
from acegen.vocabulary import SMILESVocabulary
from tensordict import TensorDict
from torch.utils.data import DataLoader, Sampler


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
    critic_training, critic_inference = create_gru_critic(len(vocabulary))
    actor_training.to(device)
    critic_training.to(device)
    actor_inference.to(device)
    critic_inference.to(device)

    print("\nCreating optimizer...")
    optimizer = torch.optim.Adam(actor_training.parameters(), lr=cfg.lr)

    # Handle wandb init
    if cfg.wandb_key:
        mode = "online"
        wandb.login(key=str(cfg.wandb_key))
    else:
        mode = "disabled"

    with wandb.init(
        project=cfg.experiment_name,
        name=cfg.agent_name + "_pretrain",
        mode=mode,
    ):

        # Calculate number of parameters
        num_params = sum(
            param.numel() for param in actor_training.policy_net.parameters()
        )
        print(f"Number of policy parameters {num_params}")

        print("\nStarting pretraining...")
        for epoch in range(1, cfg.epochs):
            dataloader.sampler.set_epoch(epoch)

            with tqdm(enumerate(dataloader), total=len(dataloader)) as tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                for step, batch in tepoch:

                    import ipdb

                    ipdb.set_trace()
                    # Optimization step
                    batch = actor_training(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()


if __name__ == "__main__":
    main()
