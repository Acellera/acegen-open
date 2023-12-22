from typing import Callable, Union

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import set_interaction_type as set_exploration_type
from tensordict.tensordict import TensorDictBase
from torchrl.collectors import RandomPolicy
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, step_mdp


def sample_completed_smiles(
    environment: EnvBase,
    policy: Union[TensorDictModule, Callable[[TensorDictBase], TensorDictBase]] = None,
    max_length: int = 100,
    end_of_episode_key: str = "done",
    exploration_type: ExplorationType = ExplorationType.RANDOM,
):
    """Samples a batch of SMILES strings from the environment.

    The SMILES strings are generated using the provided policy. Padding is used to handle
    variable length of SMILES strings.

    Args:
        environment (EnvBase): Environment to sample from.
        policy (Callable): Policy to be executed in the environment.
        Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
        If ``None`` is provided, the policy used will be a
        :class:`~torchrl.collectors.RandomPolicy` instance with the environment
        ``action_spec``.
        max_length (int, optional): Maximum length of SMILES. Defaults to 100.
        end_of_episode_key (str, optional): Key in the environment ``TensorDict`` that
        indicates the end of an episode. Defaults to "done".
        exploration_type (ExplorationType, optional): Exploration type to use. Defaults to
        :class:`~torchrl.envs.utils.ExplorationType.RANDOM`.
    """
    initial_observation = environment.reset()

    # Check that the initial observation contains the keys required by the policy
    if policy:
        for key in policy.in_keys:
            if key not in initial_observation.keys():
                raise ValueError(
                    f"Key {key}, required by the policy, is missing in the provided initial_observation."
                )
    else:
        policy = RandomPolicy(environment.action_spec)

    tensordict_device = initial_observation.device
    policy_device = policy.device
    batch_size = initial_observation.batch_size
    initial_observation = initial_observation.to(policy_device)

    # Reset environment
    tensordict_ = initial_observation
    finished = torch.zeros(batch_size, dtype=torch.bool).unsqueeze(-1).to(policy_device)

    tensordicts = []
    with set_exploration_type(exploration_type):
        for _ in range(max_length):

            # Mask out finished environments
            mask = torch.logical_not(finished)
            tensordict_.set("mask", mask)

            # Execute policy
            policy(tensordict_)

            # Extend list of tensordicts
            tensordicts.append(tensordict_)

            # Step forward in the environment
            tensordict_ = environment.step(tensordict_)

            # Step forward in the environment
            tensordict_ = step_mdp(
                tensordict_,
                keep_other=True,
                exclude_action=True,
                exclude_reward=True,
            )

            # Update finished
            finished = torch.ge(finished + tensordict_.get(end_of_episode_key), 1)

            if finished.all():
                break

    stacked_tensordicts = torch.stack(tensordicts, dim=-1)
    stacked_tensordicts = stacked_tensordicts.to(tensordict_device)

    return stacked_tensordicts
