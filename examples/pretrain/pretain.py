#!/usr/bin/env python3
"""
Pretrain a GRU or LSTM model.
"""

import datetime
import logging
import os
from pathlib import Path

import numpy as np

import pytorchrl as prl
import torch
import wandb

from acegen.de_novo_design.ppo.dataset import DeNovoDataset, load_dataset
from acegen.de_novo_design.ppo.train_rnn_model import get_args
from acegen.networks.lstm import LstmNet
from acegen.rl_environments.de_novo.generative_chemistry_env_factory import (
    de_novo_train_env_factory,
)
from acegen.rl_environments.vocabulary import ReinventVocabulary
from pytorchrl.agent.actors import OnPolicyActor
from pytorchrl.agent.env import VecEnv
from pytorchrl.utils import save_argparse
from rdkit import Chem
from torch.distributed import barrier, destroy_process_group, init_process_group

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Wraps the actor into a module for DDP
class Model(Module):
    def __init__(self, actor, device):
        super().__init__()

        self.feature_extractor = actor.policy_net.feature_extractor
        self.embedding = actor.policy_net.memory_net._embedding
        self.memory_net = actor.policy_net.memory_net._rnn
        self.dist = actor.policy_net.dist
        self.evaluator = self.dist.evaluate_pred
        self.device = device

    def forward(self, batch):

        # Prepare batch.
        seqs = batch.long()
        seqs = torch.transpose(
            seqs, dim0=0, dim1=1
        )  # Transpose because LSTM wants seqs = (seq_length, batch_size)
        seqs = seqs.to(device)

        # Prediction
        features = self.feature_extractor(seqs)
        features = self.embedding(features)
        features, _ = self.memory_net(features)
        logp_action, _, _ = self.evaluator(features[:-1, :], seqs[1:, :])

        # Loss
        mask = (seqs[1:, :] != 0).float()  # Mask padding
        loss = (-logp_action.squeeze(-1) * mask).sum(0).mean()
        # loss = -logp_action.squeeze(-1).sum(0).mean()

        return loss


def decrease_learning_rate(optimizer, decrease_by=0.01):
    """Multiplies the learning rate of the optimizer by 1 - decrease_by"""
    for param_group in optimizer.param_groups:
        param_group["lr"] *= 1 - decrease_by


def is_valid_smile(smile):
    """Returns true is smile is syntactically valid."""
    mol = Chem.MolFromSmiles(smile)
    return mol is not None


def print_master(msg):
    barrier()
    if int(os.environ["RANK"]) == 0:
        print(msg)
        logging.info(msg)
    barrier()


if __name__ == "__main__":

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

    # Parse configuration file
    args = get_args()
    if master:
        os.makedirs(args.pretrain_log_dir, exist_ok=True)
        save_argparse(args, str(Path(args.pretrain_log_dir, "conf_rnn.yaml")), [])

    # Create a checkpoint
    pretrained_ckpt = {}
    pretrained_ckpt["max_sequence_length"] = args.pretrain_max_smile_length

    # Create vocabularies
    if not os.path.exists(f"{args.pretrain_log_dir}/pretrained_ckpt.prior"):
        # NOTE: this has to be done on the master process not to blow up the memory
        if master:
            if not args.pretrainingset_path or not os.path.exists(
                args.pretrainingset_path
            ):
                raise RuntimeError("The provided pretrainingset_path is not valid!")
            print("\nConstructing vocabulary...")
            vocabulary = ReinventVocabulary.from_list(
                load_dataset(args.pretrainingset_path)
            )
            pretrained_ckpt["vocabulary"] = vocabulary
            torch.save(
                pretrained_ckpt, f"{args.pretrain_log_dir}/pretrained_ckpt.prior"
            )
    barrier()

    # Load the vocabularies
    print_master(
        f"Loading vocabularies from {args.pretrain_log_dir}/pretrained_ckpt.prior"
    )
    pretrained_ckpt = torch.load(f"{args.pretrain_log_dir}/pretrained_ckpt.prior")
    vocabulary = pretrained_ckpt["vocabulary"]

    print_master("\nPreparing dataset and dataloader...")
    # Precompute the dataset on a single process
    if master:
        dataset = DeNovoDataset(
            cache_path=args.dataset_log_dir or args.pretrain_log_dir,
            dataset_path=args.pretrainingset_path,
            vocabulary=vocabulary,
            randomize_smiles=args.randomize_pretrain_smiles,
        )
        dataset_tokens = len(dataset.mmaps["smiles_data"])
        print(f"Dataset has {dataset_tokens} tokens in total")
        pretrained_ckpt["num_dataset_tokens"] = dataset_tokens
    barrier()
    dataset = DeNovoDataset(
        cache_path=args.dataset_log_dir or args.pretrain_log_dir,
        dataset_path=args.pretrainingset_path,
        vocabulary=vocabulary,
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.pretrain_batch_size,
        sampler=DistributedSampler(dataset),
        shuffle=False,  # Needs to be False with DistributedSampler
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Define env
    test_env, action_space, obs_space = VecEnv.create_factory(
        env_fn=de_novo_train_env_factory,
        env_kwargs={
            "scoring_function": lambda a: {"reward": 1.0},
            "vocabulary": vocabulary,
            "smiles_max_length": args.pretrain_max_smile_length,
        },
        vec_env_size=1,
    )
    env = test_env(device)

    # Define model
    feature_extractor_kwargs = {}
    recurrent_net_kwargs = {
        "vocabulary_size": len(vocabulary),
        "embedding_size": args.embeddings_size,
        "num_layers": args.lstm_num_layers,
        "layer_size": args.lstm_num_nodes_per_layer,
        "dropout": args.lstm_dropout_prob,
    }

    actor = OnPolicyActor.create_factory(
        obs_space,
        action_space,
        prl.PPO,
        feature_extractor_network=torch.nn.Identity,
        feature_extractor_kwargs={**feature_extractor_kwargs},
        recurrent_net=LstmNet,
        recurrent_net_kwargs={**recurrent_net_kwargs},
    )(device)
    pretrained_ckpt["feature_extractor_kwargs"] = feature_extractor_kwargs
    pretrained_ckpt["recurrent_net_kwargs"] = recurrent_net_kwargs

    # Create a distributed model
    model = DistributedDataParallel(Model(actor, device))

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr)

    # Handle wandb init
    if args.wandb_key:
        mode = "online"
        wandb.login(key=str(args.wandb_key))
    else:
        mode = "disabled"

    # Handle tensorboard init
    if args.tensorboard_logging:
        tb_writer = SummaryWriter(
            log_dir=Path(args.pretrain_log_dir, "tensorboard_logs")
        )
    else:
        tb_writer = None

    with wandb.init(
        project=args.experiment_name,
        name=args.agent_name + "_pretrain",
        config=args,
        mode=mode,
    ):

        # Calculate number of parameters
        num_params = sum(param.numel() for param in actor.policy_net.parameters())
        print_master(f"Number of policy parameters {num_params}")
        pretrained_ckpt["num_policy_parameters"] = num_params
        args.policy_parameters = num_params

        print_master("\nStarting pretraining...")
        for epoch in range(1, args.pretrain_epochs):
            dataloader.sampler.set_epoch(epoch)

            with tqdm(enumerate(dataloader), total=len(dataloader)) as tepoch:

                tepoch.set_description(f"Epoch {epoch}")

                for step, batch in tepoch:

                    # Optimization step
                    loss = model(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    info_dict = {}
                    total_steps = step + len(dataloader) * (epoch - 1)
                    if (
                        total_steps % args.pretrain_lr_decrease_period
                    ) == 0 and total_steps != 0:

                        # Eval mode
                        actor.eval()

                        # Decrease learning rate
                        decrease_learning_rate(
                            optimizer, decrease_by=args.pretrain_lr_decrease_value
                        )

                        # Generate a few molecules and check how many are valid
                        total_molecules = 100
                        valid_molecules = 0
                        list_molecules = []
                        list_num_tokens = []
                        list_entropy = []
                        for i in range(total_molecules):
                            obs, rhs, done = actor.actor_initial_states(env.reset())
                            molecule = "^"
                            num_tokens = 0
                            while not done:
                                with torch.no_grad():
                                    prediction = actor.get_action(
                                        obs, rhs, done, deterministic=False
                                    )
                                    (_, action, _, rhs, entropy_dist, dist) = prediction
                                obs, _, done, _ = env.step(action)
                                molecule += vocabulary.decode_token(action)
                                list_entropy.append(entropy_dist.item())
                                num_tokens += 1

                            if is_valid_smile(
                                vocabulary.remove_start_and_end_tokens(molecule)
                            ):
                                valid_molecules += 1
                            list_molecules.append(molecule)
                            list_num_tokens.append(num_tokens)

                        # Check how many are repeated
                        ratio_repeated = (
                            len(set(list_molecules)) / len(list_molecules)
                            if total_molecules > 0
                            else 0
                        )

                        # Add to info dict
                        info_dict.update(
                            {
                                "pretrain_avg_molecular_length": np.mean(
                                    list_num_tokens
                                ),
                                "pretrain_avg_entropy": np.mean(list_entropy),
                                "pretrain_valid_molecules": (
                                    valid_molecules / total_molecules
                                ),
                                "pretrain_ratio_repeated": ratio_repeated,
                            }
                        )

                        # Train mode
                        actor.train()

                    tepoch.set_postfix(loss=loss.item())

                    # Logging
                    info_dict.update({"pretrain_loss": loss.item()})
                    wandb.log(info_dict, step=total_steps)
                    if tb_writer:
                        for tag, value in info_dict.items():
                            tb_writer.add_scalar(tag, value, global_step=total_steps)

            # Save model
            pretrained_ckpt["network_weights"] = actor.state_dict()
            if master:
                torch.save(
                    pretrained_ckpt,
                    f"{args.pretrain_log_dir}/pretrained_ckpt_epoch{epoch}.prior",
                )
            barrier()

    print_master("Finished!")
    destroy_process_group()
