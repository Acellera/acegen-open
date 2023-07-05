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
    TensorDictPrimer,
)

from env import GenChemEnv
from vocabulary import DeNovoVocabulary
from utils import create_policy, create_critic, create_rhs_transform


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: "DictConfig"):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Create vocabulary from data
    vocabulary = DeNovoVocabulary.from_list("dataset")

    # Environment
    ####################################################################################################################

    # Let's use a random scoring function for now
    def dummy_scoring(smiles):
        output = {
            "reward": random.random(),
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

    actor = ProbabilisticActor(
        module=create_policy(vocabulary=vocabulary, output_size=action_spec.shape[-1]),
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
    )
    actor = actor.to(device)
    critic = create_critic(vocabulary=vocabulary, output_size=1, out_key="state_value")
    critic = critic.to(device)

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=True,
    )
    loss_module = PPOLoss(actor, critic)
    loss_module = loss_module.to(device)

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device="cuda:0",
        storing_device="cuda:0",
        max_frames_per_traj=-1,
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=0.001,
        weight_decay=0.000,
    )

    # Training loop
    ####################################################################################################################

    for batch in collector:

        print("step!")

        # batch = batch.to(device)
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
