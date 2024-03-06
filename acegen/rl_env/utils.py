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
    scoring_function: Callable = None,
    policy_sample: Union[
        TensorDictModule, Callable[[TensorDictBase], TensorDictBase]
    ] = None,
    policy_evaluate: Union[
        TensorDictModule, Callable[[TensorDictBase], TensorDictBase]
    ] = None,
    prompt: Union[str, list] = None,
    max_length: int = None,
    end_of_episode_key: str = "done",
    exploration_type: ExplorationType = ExplorationType.RANDOM,
    promptsmiles: str = None,
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
        vocabulary (SMILESVocabulary): Vocabulary to use for encoding and decoding SMILES strings,
            necessary for promptsmiles.
        scoring_function (Callable, optional): Scoring function to be used to evaluate the generated SMILES.
        policy_sample (Callable): Policy to be executed in the environment step-by_step of shape (batch_size, 1).
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
        policy_evaluate (Callable, optional): Policy to be used to evaluate the log probability of the actions.
            of shape (batch_size, time). Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
        If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
        prompt (Union[str, list], optional): SMILES string or list of SMILES strings to be used as prompts.
        max_length (int, optional): Maximum length of SMILES. Defaults to 100.
        end_of_episode_key (str, optional): Key in the environment ``TensorDict`` that
            indicates the end of an episode. Defaults to "done".
        exploration_type (ExplorationType, optional): Exploration type to use. Defaults to
            :class:`~torchrl.envs.utils.ExplorationType.RANDOM`.
        promptsmiles (str, optional): SMILES string of scaffold with attachment points or fragments seperated
            by "." with one attachment point each.
        promptsmiles_optimize (bool, optional): Optimize the prompt for the model being used.
            Defaults to True.
        promptsmiles_shuffle (bool, optional): Shuffle the selected attachmented point within the batch.
            Defaults to True.
        return_smiles_only (bool, optional): If ``True``, only the SMILES strings are returned.
            Only when not using a PrompSMILES argument. Defaults to False.
    """
    env_device = environment.device
    initial_observation = environment.reset()
    batch_size = initial_observation.batch_size
    max_length = max_length or environment.max_length
    if policy_sample:
        # Check that the initial observation contains the keys required by the policy
        for key in policy_sample.in_keys:
            if key not in initial_observation.keys():
                raise ValueError(
                    f"Key {key}, required by the policy, is missing in the provided initial_observation."
                )
        policy_device = policy_sample.device
    else:
        policy_evaluate = RandomPolicy(environment.action_spec)
        policy_device = env_device

    # ----- Insertion of PROMPTSMILES -----

    if promptsmiles:
        if not _has_promptsmiles:
            raise RuntimeError(
                "PromptSMILES library not found, please install with pip install promptsmiles ."
            ) from PROMPTSMILES_ERR

        # Setup sample fn passed to promptsmiles
        sample_fn = partial(
            generate_complete_smiles,
            environment=environment,
            vocabulary=vocabulary,
            policy_sample=policy_sample,
            policy_evaluate=policy_evaluate,
            promptsmiles=None,
            max_length=max_length,
            end_of_episode_key=end_of_episode_key,
            exploration_type=exploration_type,
            return_smiles_only=True,
        )
        # Setup evaluate fn passed to promptsmiles

        if policy_evaluate is None:
            raise ValueError(
                "policy_evaluate parameter must be provided when using promptsmiles."
            )

        evaluate_fn = partial(
            _get_log_prob,
            policy=policy_evaluate,
            vocabulary=vocabulary,
            max_length=max_length,
        )
        # Split fragments into a list if there are multiple
        promptsmiles = promptsmiles.split(".")
        if len(promptsmiles) == 1:
            promptsmiles = promptsmiles[0]

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
                return_all=True,
            )

        if isinstance(promptsmiles, list):
            # We are linking fragments
            PS = FragmentLinker(
                fragments=promptsmiles,
                batch_size=batch_size[0],
                sample_fn=sample_fn,
                evaluate_fn=evaluate_fn,
                batch_prompts=True,
                optimize_prompts=promptsmiles_optimize,
                shuffle=promptsmiles_shuffle,
                return_all=True,
            )

        smiles = PS.sample()

        # Encode all smiles
        enc_smiles = []
        for promptiteration in smiles:
            tokens = [torch.tensor(vocabulary.encode(smi)) for smi in promptiteration]
            enc_smiles.append(
                torch.vstack(
                    [
                        torch.nn.functional.pad(
                            tok, (0, max_length + 1 - tok.size()[0])
                        )
                        for tok in tokens
                    ]
                )
            )

        # Create output_data format from final completed SMILES
        output_data = smiles_to_tensordict(
            enc_smiles[-1], mask_value=0, device=env_device
        )

        # Also append all intermediate steps, skip the start token to be the same as action
        output_data.set(
            "promptsmiles",
            torch.stack([e[:, 1:] for e in enc_smiles], dim=-1).to(env_device),
        )
        return output_data

    # ----------------------------------------

    else:

        if prompt:
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size[0]
            tokens = [
                torch.tensor(vocabulary.encode(smi, with_start=False, with_end=False))
                for smi in prompt
            ]
            enc_prompts = torch.vstack(
                [
                    torch.nn.functional.pad(tok, (0, max_length + 1 - tok.size()[0]))
                    for tok in tokens
                ]
            ).to(policy_device)

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
                    if prompt:
                        enforce_mask = enc_prompts[:, _] != 0

                    # Execute policy
                    tensordict_ = tensordict_.to(policy_device)
                    policy_sample(tensordict_)
                    tensordict_ = tensordict_.to(env_device)

                    # Enforce prompt
                    if prompt:
                        new_action = (~enforce_mask * tensordict_.get("action")) + (
                            enforce_mask * enc_prompts[:, _]
                        ).long()
                        tensordict_.set("action", new_action)

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
        output_data.refine_names(..., "time")

    if return_smiles_only:
        smiles = output_data.get("action").cpu()
        smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles]
        return smiles_str
    else:

        # Compute rewards
        done = output_data.get("done").squeeze(-1)
        smiles = output_data.get("action").cpu()
        smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles]
        output_data["reward"][done] = torch.tensor(
            scoring_function(smiles_str), device=output_data.device
        ).unsqueeze(-1)

        # For promptsmiles, update the action key
        if promptsmiles:

            # if cfg.get("promptsmiles_multi"):
            #     print(
            #         NotImplementedError(
            #             "promptsmiles with multi updates is not implemented yet, running with single update."
            #         )
            #     )

            # Depending on fragment or scaffold
            if "." in promptsmiles:
                ps_idx = 0
            else:
                ps_idx = -1
            output_data.set("action", output_data.get("promptsmiles")[:, :, ps_idx])

            # # TODO: why do we re-set the action but not the observation, next observation etc?
            # # TODO: Also, do "action" and "promptsmiles" have the same length? because otherwise we should also change
            # # TODO: Maybe it is easier to do data = smiles_to_tensordict(data.get("promptsmiles")[:, :, ps_idx],
            #  data.get("reward"))
            # # the position of the reward and the done/terminated/truncated flags
            # data.set("action", data.get("promptsmiles")[:, :, ps_idx])
            #
            # # For transformers-based policies
            # start_token = torch.full(
            #     (data.batch_size[0], 1), vocabulary.start_token_index, dtype=torch.long
            # ).to(data.device)
            # data.set(
            #     "sequence",
            #     torch.cat(
            #         [start_token, data.get("promptsmiles")[:, :-1, ps_idx]], dim=1
            #     ),
            # )
            # data.set(("next", "sequence"), data.get("promptsmiles")[:, :, ps_idx])

        return output_data


def _get_log_prob(
    smiles: list,
    policy: Union[TensorDictModule, Callable[[TensorDictBase], TensorDictBase]],
    vocabulary: Vocabulary,
    max_length: int,
):
    # Convert SMILES to TensorDict
    tokens = [torch.tensor(vocabulary.encode(smi)) for smi in smiles]
    enc_smiles = torch.vstack(
        [
            torch.nn.functional.pad(tok, (0, max_length - tok.size()[0]))
            for tok in tokens
        ]
    )
    data = smiles_to_tensordict(enc_smiles, mask_value=0, device=policy.device)
    actions = data.get("action").clone()

    # For transformers-based policies
    data.set("sequence", data.get("observation"))

    policy_in = data.select(*policy.in_keys, strict=False)
    log_prob = policy.get_dist(policy_in).log_prob(actions)
    log_prob = log_prob.sum(-1)
    return log_prob.cpu()
