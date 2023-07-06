import os.path

import rdkit
import hydra
import torch
import random

from torchrl.objectives import PPOLoss
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.envs import (
    ParallelEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
)

from env import GenChemEnv
from vocabulary import DeNovoVocabulary
from utils import create_model, create_rhs_transform, partially_load_checkpoint


# TODO: load checkpoint partially for value net
# TODO: save checkpoints to avoid adapting every time
# TODO: add training logging
# TODO: add smiles logging


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: "DictConfig"):

    device = torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")

    ####################################################################################################################

    vocabulary = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocabulary.prior"))

    # Environment
    ####################################################################################################################

    # Let's use a basic scoring function that gives a reward of 1.0 if the SMILES is valid and 0.0 otherwise.
    def dummy_scoring(smiles):
        mol = rdkit.Chem.MolFromSmiles(smiles)
        output = {
            "reward": 1.0 if mol else 0.0,
            "valid_smile": True,
        }
        return output

    env_kwargs = {"scoring_function": dummy_scoring, "vocabulary": vocabulary}

    def create_transformed_env():
        env = GymWrapper(GenChemEnv(**env_kwargs), categorical_action_encoding=True)
        env = TransformedEnv(env)
        env.append_transform(create_rhs_transform())
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        env = ParallelEnv(
            create_env_fn=create_transformed_env,
            num_workers=num_workers,
        )
        return env

    test_env = GymWrapper(GenChemEnv(**env_kwargs))
    action_spec = test_env.action_spec
    observation_spec = test_env.observation_spec

    # Models
    ####################################################################################################################

    actor_model = create_model(vocabulary=vocabulary, output_size=action_spec.shape[-1])
    actor_model.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "actor.prior")))
    actor = ProbabilisticActor(
        module=actor_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    actor = actor.to(device)
    critic = create_model(vocabulary=vocabulary, output_size=1, out_key="state_value")
    critic.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "critic.prior")))
    critic = critic.to(device)

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=True,
        shifted=True,
    )
    loss_module = PPOLoss(actor, critic, critic_coef=cfg.critic_coef, entropy_coef=cfg.entropy_coef)
    loss_module = loss_module.to(device)

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        max_frames_per_traj=-1,
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=0.000,
    )

    # Training loop
    ####################################################################################################################

    for batch in collector:

        print("step!")

        batch = batch.to(device)
        with torch.no_grad():
            batch = adv_module(batch)

        loss = loss_module(batch)
        loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

        # Backward pass
        loss_sum.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=0.5)

        optim.step()
        optim.zero_grad()

    collector.shutdown()


if __name__ == "__main__":
    main()
