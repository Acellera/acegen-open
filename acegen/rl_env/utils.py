from functools import partial
from typing import Callable, Union

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import set_interaction_type as set_exploration_type
from tensordict.tensordict import TensorDictBase
from torchrl.collectors import RandomPolicy
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, step_mdp

from acegen.data.utils import smiles_to_tensordict
from acegen.vocabulary import SMILESVocabulary, Vocabulary

try:
    from promptsmiles import FragmentLinker, ScaffoldDecorator

    _has_promptsmiles = True
except ImportError as err:
    _has_promptsmiles = False
    PROMPTSMILES_ERR = err


@torch.no_grad()
def generate_complete_smiles(
    environment: EnvBase,
    vocabulary: SMILESVocabulary,
    policy: Union[TensorDictModule, Callable[[TensorDictBase], TensorDictBase]] = None,
    prompt: Union[str, list] = None,
    max_length: int = None,
    end_of_episode_key: str = "done",
    exploration_type: ExplorationType = ExplorationType.RANDOM,
    promptsmiles: Union[str, list] = None,
    promptsmiles_optimize: bool = True,
    promptsmiles_shuffle: bool = True,
    return_smiles_only: bool = False,
    **kwargs,
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
    env_device = environment.device
    initial_observation = environment.reset()
    batch_size = initial_observation.batch_size
    max_length = max_length or environment.max_length

    # ----- Insertion of PROMPTSMILES -----
    if promptsmiles:
        if not _has_promptsmiles:
            raise RuntimeError(
                "PromptSMILES library not found, please install with pip install promptsmiles ."
            ) from PROMPTSMILES_ERR

        sample_fn = partial(
            generate_complete_smiles,
            environment=environment,
            vocabulary=vocabulary,
            policy=policy,
            promptsmiles=None,
            max_length=max_length,
            end_of_episode_key=end_of_episode_key,
            exploration_type=exploration_type,
            return_smiles_only=True,
        )
        evaluate_fn = partial(
            get_log_prob,
            policy=policy,
            vocabulary=vocabulary,
            max_length=max_length
            )

        if isinstance(promptsmiles, str):
            # We are decorating a Scaffold
            PS = ScaffoldDecorator(
                scaffold=promptsmiles,
                batch_size=batch_size[0],
                sample_fn=sample_fn,
                evaluate_fn=evaluate_fn,
                batch_prompts=True,
                optimize_prompts=promptsmiles_optimize,
                shuffle=promptsmiles_shuffle,
                return_all=False,  # True -> [[SMILES@1], [SMILES@2], ...], [[NLLS@1], [NLLS@2], ...]
            )
            smiles = PS.sample()
            tokens = [torch.tensor(vocabulary.encode(smi)) for smi in smiles]
            enc_smiles = torch.vstack([torch.nn.functional.pad(tok, (0, max_length+1-tok.size()[0])) for tok in tokens])
            output_data = smiles_to_tensordict(enc_smiles, mask_value=0)
            return output_data

        if isinstance(promptsmiles, list):
            # We are linking fragments
            PS = FragmenterLinker(
                fragments=promptsmiles,
                batch_size=batch_size[0],
                sample_fn=sample_fn,
                evaluate_fn=evaluate_fn,
                batch_prompts=True,
                optimize_prompts=promptsmiles_optimize,
                shuffle=promptsmiles_shuffle,
                return_all=False,  # True -> [[SMILES@1], [SMILES@2], ...], [[NLLS@1], [NLLS@2], ...]
            )
            smiles = PS.sample()
            tokens = [torch.tensor(vocabulary.encode(smi)) for smi in smiles]
            enc_smiles = torch.vstack([torch.nn.functional.pad(tok, (0, max_length+1-tok.size()[0])) for tok in tokens])
            output_data = smiles_to_tensordict(enc_smiles, mask_value=0)
            return output_data

    # ----------------------------------------

    else:
        if policy:
            # Check that the initial observation contains the keys required by the policy
            for key in policy.in_keys:
                if key not in initial_observation.keys():
                    raise ValueError(
                        f"Key {key}, required by the policy, is missing in the provided initial_observation."
                    )
            policy_device = policy.device
        else:
            policy = RandomPolicy(environment.action_spec)
            policy_device = env_device

        if prompt:
            import pdb; pdb.set_trace()
            if isinstance(prompt, str): prompt = [prompt]*batch_size[0]
            tokens = [torch.tensor(vocabulary.encode(smi, with_end=True)) for smi in prompt]
            enc_prompts = torch.vstack([torch.nn.functional.pad(tok, (0, max_length+1-tok.size()[0])) for tok in tokens])
            enc_prompts = smiles_to_tensordict(enc_prompts, mask_value=0, device=policy_device)
            enc_prompts.set('is_init', torch.zeros_like(enc_prompts.get('done')))
            policy(enc_prompts)
            done_prompts = enc_prompts.get(('next', 'done')).squeeze(-1)
            done_state = torch.roll(done_prompts.clone(), shifts=-1, dims=-1)
            initial_observation.update(enc_prompts.get('next').select('observation')[done_state])
            initial_observation.update(enc_prompts.get('next').select('recurrent_state_actor')[done_state])
        
        initial_observation = initial_observation.to(policy_device)
        tensordict_ = initial_observation
        finished = (
            torch.zeros(batch_size, dtype=torch.bool).unsqueeze(-1).to(policy_device)
        )

        tensordicts = []
        with set_exploration_type(exploration_type):
            for _ in range(max_length):

                if not finished.all():

                    # Define mask tensors
                    tensordict_.set("mask", torch.ones_like(finished))
                    tensordict_.set(("next", "mask"), torch.ones_like(finished))

                    # Execute policy
                    tensordict_ = tensordict_.to(policy_device)
                    policy(tensordict_)
                    tensordict_ = tensordict_.to(env_device)

                    # Step forward in the environment
                    tensordict_ = environment.step(tensordict_)

                    # Mask out finished environments
                    if finished.any():
                        tensordict_.masked_fill_(finished.squeeze(), 0)

                    # Extend list of tensordicts
                    tensordicts.append(tensordict_.clone())

                    # Step forward in the environment
                    tensordict_ = step_mdp(
                        tensordict_,
                        keep_other=True,
                        exclude_action=True,
                        exclude_reward=True,
                    )

                    # Update finished
                    finished = torch.ge(
                        finished + tensordict_.get(end_of_episode_key), 1
                    )

                else:
                    tensordicts.append(torch.zeros_like(tensordicts[-1]))

        # If after max_length steps the SMILES are not finished, truncate
        if not finished.all():
            tensordicts[-1][("next", "truncated")] = ~finished.clone()
            tensordicts[-1][("next", "done")] = ~finished.clone()

        output_data = torch.stack(tensordicts, dim=-1).contiguous()
        #import pdb; pdb.set_trace()
        output_data.refine_names(..., "time")

    if return_smiles_only:
        smiles = output_data.select("action").cpu()
        smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles["action"]]
        smiles_str = [p + s for p, s in zip(prompt, smiles_str)]
        return smiles_str
    else:
        return output_data


def get_log_prob(
    smiles: list,
    policy: Union[TensorDictModule, Callable[[TensorDictBase], TensorDictBase]],
    vocabulary: Vocabulary,
    max_length: int,
):
    # Convert SMILES to TensorDict
    tokens = [torch.tensor(vocabulary.encode(smi)) for smi in smiles]
    enc_smiles = torch.vstack([torch.nn.functional.pad(tok, (0, max_length-tok.size()[0])) for tok in tokens])
    data = smiles_to_tensordict(enc_smiles, mask_value=0)
    data.set('is_init', torch.zeros_like(data.get('done')))

    actions = data.get("action").clone()

    # For transformers-based policies
    data.set("sequence", data.get("observation"))

    policy_in = data.select(*policy.in_keys, strict=False)
    log_prob = policy.get_dist(policy_in).log_prob(actions)
    log_prob = log_prob.sum(-1)
    return log_prob
