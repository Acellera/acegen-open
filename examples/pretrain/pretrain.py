import random

import re
from pathlib import Path

import hydra
import numpy as np
import torch
from acegen.data import load_dataset, remove_duplicates, SMILESDataset
from acegen.models import create_gru_actor
from acegen.rl_env import generate_complete_smiles, SMILESEnv
from acegen.vocabulary import SMILESVocabulary
from rdkit import Chem
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torchrl.envs import InitTracker, TensorDictPrimer, TransformedEnv
from torchrl.record.loggers import get_logger
from tqdm import tqdm


class Tokenizer:
    def __init__(self, start_token: str = "GO", end_token: str = "EOS"):
        self.start_token = start_token
        self.end_token = end_token

    @staticmethod
    def replace_halogen(string: str) -> str:
        """Regex to replace Br and Cl with single letters."""
        br = re.compile("Br")
        cl = re.compile("Cl")
        string = br.sub("R", string)
        string = cl.sub("L", string)
        return string

    def tokenize(self, smiles: str) -> list[str]:
        regex = "(\[[^\[\]]{1,6}\])"
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = [self.start_token]
        for char in char_list:
            if char.startswith("["):
                tokenized.append(char)
            else:
                [tokenized.append(unit) for unit in list(char)]
        tokenized.append(self.end_token)
        return tokenized


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    # Set seeds
    seed = cfg.seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    device = f"cuda:0" if torch.cuda.device_count() > 1 else "cpu"

    cfg.train_dataset_path = (
        Path(__file__).resolve().parent.parent.parent / "priors" / "smiles_test_set"
    )

    print("\nConstructing vocabulary...")
    vocabulary = SMILESVocabulary.create_from_smiles(
        load_dataset(cfg.train_dataset_path),
        tokenizer=Tokenizer(),
    )
    save_path = Path(cfg.model_log_dir) / "vocabulary.pkl"
    torch.save(vocabulary.state_dict(), save_path)

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
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    print("\nCreating model...")
    actor_training, actor_inference = create_gru_actor(
        len(vocabulary), embedding_size=128
    )
    actor_training.to(device)
    actor_inference.to(device)

    print("\nCreating test environment...")
    # Create a transform to populate initial tensordict with rnn recurrent states equal to 0.0
    primers = actor_training.rnn_spec.expand(cfg.num_test_smiles)
    rhs_primer = TensorDictPrimer(primers)
    test_env = SMILESEnv(
        start_token=vocabulary.vocab[vocabulary.start_token],
        end_token=vocabulary.vocab[vocabulary.end_token],
        length_vocabulary=len(vocabulary),
        batch_size=100,
        device=device,
    )
    test_env = TransformedEnv(test_env)
    test_env.append_transform(InitTracker())
    test_env.append_transform(rhs_primer)

    print("\nCreating test scoring function...")

    def valid_smiles(smiles_list):
        result_tensor = torch.zeros(len(smiles_list))
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                result_tensor[i] = 1.0
        return result_tensor

    print("\nCreating optimizer...")
    actor_optimizer = torch.optim.Adam(actor_training.parameters(), lr=cfg.lr)

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

                # smiles = [vocabulary.decode(smi.cpu().numpy(), ignore_indices=[-1]) for smi in batch]
                # num_valid = valid_smiles(smiles).sum()

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

                # Loss
                loss_actor = (
                    (-td_batch.get("sample_log_prob").squeeze(-1) * mask).sum(0).mean()
                )

                # Backward pass
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()
                actor_losses[step] = loss_actor.item()

            # Generate test smiles
            smiles = generate_complete_smiles(test_env, actor_inference, max_length=100)
            num_valid = valid_smiles(
                [vocabulary.decode(smi.cpu().numpy()) for smi in smiles.get("action")]
            ).sum()
            unique_smiles = remove_duplicates(smiles, key="action")

            # Log
            if logger:
                logger.log_scalar("loss_actor", actor_losses.mean())
                logger.log_scalar("num_test_valid_smiles", num_valid)
                logger.log_scalar("num_test_unique_smiles", len(unique_smiles))

        save_path = Path(cfg.model_log_dir) / f"pretrained_actor_epoch_{epoch}.pt"
        torch.save(actor_training.state_dict(), save_path)


if __name__ == "__main__":
    main()
