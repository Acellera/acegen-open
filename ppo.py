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
        env.append_transform(TensorDictPrimer())
        return env

    def create_env_fn(num_workers=2):
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
        gamma=0.99,
        lmbda=0.95,
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
        frames_per_batch=64,
        total_frames=128,
        device="cpu",
        storing_device="cpu",
        max_frames_per_traj=-1,
    )

    for batch in collector:

        batch = batch.to(device)
        with torch.no_grad():
            batch = adv_module(batch)
        batch = batch.reshape(-1)
        loss = loss_module(batch)  # TypeError: expected Tensor as element 0 in argument 2, but got NoneType

    collector.shutdown()


if __name__ == "__main__":
    main()
