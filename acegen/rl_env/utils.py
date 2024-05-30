import warnings
from functools import partial
from typing import Callable, Union

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.probabilistic import set_interaction_type as set_exploration_type
from tensordict.tensordict import TensorDictBase
from tensordict.utils import remove_duplicates as rdups
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
    end_of_episode_key: str = "done",
    exploration_type: ExplorationType = ExplorationType.RANDOM,
    promptsmiles: str = None,
    promptsmiles_optimize: bool = True,
    promptsmiles_shuffle: bool = True,
    promptsmiles_multi: bool = False,
    promptsmiles_scan: bool = False,
    remove_duplicates: bool = False,
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
        promptsmiles_multi (bool, optional): Return all promptsmiles iterations. Resulting in multiple updates
            per SMILES. Defaults to False.
        promptsmiles_scan (bool, optional): Fragment linking mode only, scan through linker positions.
            Defaults to False.
        remove_duplicates (bool, optional): Remove duplicate SMILES strings from the output. Defaults to False.
        return_smiles_only (bool, optional): If ``True``, only the SMILES strings are returned.
            Only when not using a PrompSMILES argument. Defaults to False.
    """
    env_device = environment.device
    initial_observation = environment.reset()
    batch_size = initial_observation.batch_size
    max_length = environment.max_length
    if policy_sample:
        # Check that the initial observation contains the keys required by the policy
        for key in policy_sample.in_keys:
            if key not in initial_observation.keys():
                raise ValueError(
                    f"Key {key}, required by the policy, is missing in the provided initial_observation."
                )
        policy_device = policy_sample.device
    else:
        policy_sample = RandomPolicy(environment.action_spec)
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
                scan=promptsmiles_scan,
                return_all=True,
            )

        smiles = PS.sample()

        # Encode all smiles
        enc_smiles = []
        failed_encodings = set()
        for promptiteration in smiles[::-1]:
            tokens = []
            for i, smi in enumerate(promptiteration):
                try:
                    tokens.append(torch.tensor(vocabulary.encode(smi)))
                # PromptSMILES may generate tokens outside of the working vocabulary
                except KeyError as e:
                    warnings.warn(
                        f"Failed to encode {smi} with error {e}: ignoring invalid prompt"
                    )
                    failed_encodings.add(i)
                    tokens.append(torch.tensor(vocabulary.encode("")))

            enc_smiles.append(
                torch.vstack(
                    [
                        # Pad with -1 in case the end token is already 0
                        torch.nn.functional.pad(
                            tok, (0, max_length + 1 - tok.size()[0]), value=-1
                        )
                        for tok in tokens
                    ]
                )
            )
        # Put back to correct order
        enc_smiles = enc_smiles[::-1]

        if promptsmiles_multi:
            # Stack all promptsmiles iterations
            promptiterations = []
            for enc_smi in enc_smiles:
                _output_data = smiles_to_tensordict(
                    enc_smi,
                    mask_value=-1,
                    replace_mask_value=vocabulary.end_token_index,
                    device=env_device,
                )
                # Add final complete smiles for logging and scoring
                _output_data.set(
                    "promptsmiles",
                    enc_smiles[-1][:, :-1].to(env_device),
                )
                # Fix failed encodings
                if failed_encodings:
                    failed_mask = torch.zeros(batch_size[0], dtype=torch.bool).to(
                        env_device
                    )
                    failed_mask[list(failed_encodings)] = True
                    _output_data["action"].masked_fill_(
                        failed_mask.unsqueeze(-1).expand_as(_output_data["action"]), 0
                    )
                promptiterations.append(_output_data)
            output_data = torch.cat(promptiterations, dim=0).contiguous()
        else:
            # Re-compute tensordict
            if isinstance(promptsmiles, list):
                # Fragment linking, 0 is first and only position where tokens are sampled
                ps_idx = 0
            else:
                # Scaffold decoration, final completed smiles (all attachment points)
                ps_idx = -1

            # Create tensordicts
            output_data = smiles_to_tensordict(
                enc_smiles[ps_idx],
                mask_value=-1,
                replace_mask_value=vocabulary.end_token_index,
                device=env_device,
            )

            # Fix failed encodings
            if failed_encodings:
                failed_mask = torch.zeros(batch_size[0], dtype=torch.bool).to(
                    env_device
                )
                failed_mask[list(failed_encodings)] = True
                output_data["action"].masked_fill_(
                    failed_mask.unsqueeze(-1).expand_as(output_data["action"]), 0
                )

            # Add final completed promptsmiles for logging and scoring
            output_data.set(
                "promptsmiles",
                enc_smiles[-1][:, :-1].to(env_device),
            )

        if remove_duplicates:
            output_data = rdups(output_data, key="action")

        # For transformers-based policies
        output_data.set("sequence", output_data.get("observation"))
        output_data.set("sequence_mask", output_data.get("mask"))
        output_data.set(("next", "sequence"), output_data.get(("next", "observation")))
        output_data.set(("next", "sequence_mask"), output_data.get(("next", "mask")))

        # Compute rewards
        if scoring_function:
            smiles = output_data.get("promptsmiles").cpu()
            next_output_data = output_data.get("next")
            done = next_output_data.get("done").squeeze(-1)
            smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles]
            next_output_data["reward"][done] = torch.tensor(
                scoring_function(smiles_str), device=output_data.device
            ).unsqueeze(-1)

        # Recompute policy log_prob
        if (
            "sample_log_prob" not in output_data.keys()
        ):  # Not ideal, because we do an extra forward pass, but it works
            with torch.no_grad():
                output_data.set(
                    "sample_log_prob",
                    policy_evaluate.get_dist(output_data.clone()).log_prob(
                        output_data["action"]
                    ),
                )

        return output_data

    # ----------------------------------------

    else:

        failed_encodings = []
        if prompt:
            if isinstance(prompt, str):
                prompt = [prompt] * batch_size[0]
            # Encode the prompt(s)
            tokens = []
            for i, smi in enumerate(prompt):
                try:
                    ts = vocabulary.encode(smi, with_start=False, with_end=False)
                    if len(ts) > max_length:
                        raise ValueError(
                            f"Prompt {smi} is longer than max_length {max_length}."
                        )
                    else:
                        tokens.append(torch.tensor(ts))
                # Prompt may contain tokens outside of the working vocabulary
                except KeyError:
                    failed_encodings.append(i)
                    tokens.append(
                        torch.tensor(
                            vocabulary.encode("", with_start=False, with_end=False)
                        )
                    )
            # If they all fail, raise an error
            if len(failed_encodings) == batch_size[0]:
                raise RuntimeError(
                    "All prompts failed to encode, ensure SMILES fits the vocabulary."
                )
            enc_prompts = torch.vstack(
                [
                    # Pad since prompts do not have end tokens
                    torch.nn.functional.pad(
                        tok,
                        (0, max_length + 1 - tok.size()[0]),
                        value=vocabulary.end_token_index,
                    )
                    for tok in tokens
                ]
            ).to(policy_device)

        initial_observation = initial_observation.to(policy_device)
        tensordict_ = initial_observation
        finished = (
            torch.zeros(batch_size, dtype=torch.bool).unsqueeze(-1).to(policy_device)
        )
        if failed_encodings:
            finished[failed_encodings] = True

        tensordicts = []
        with set_exploration_type(exploration_type):
            for _ in range(max_length):

                if not finished.all():
                    # Define mask tensors
                    tensordict_.set("mask", torch.ones_like(finished))
                    tensordict_.set(("next", "mask"), torch.ones_like(finished))
                    if prompt:
                        enforce_mask = enc_prompts[:, _] != vocabulary.end_token_index

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
        # Replace failed encodings with original prompt for PromptSMILES
        for i in failed_encodings:
            smiles_str[i] = prompt[i]
        return smiles_str

    else:

        # For transformers-based policies
        output_data.set("sequence", output_data.get("observation"))
        output_data.set("sequence_mask", output_data.get("mask"))
        output_data.set(("next", "sequence"), output_data.get(("next", "observation")))
        output_data.set(("next", "sequence_mask"), output_data.get(("next", "mask")))

        if remove_duplicates:
            output_data = rdups(output_data, key="action")

        # Compute rewards
        if scoring_function:
            smiles = output_data.get("action").cpu()
            next_output_data = output_data.get("next")
            done = next_output_data.get("done").squeeze(-1)
            smiles_str = [vocabulary.decode(smi.numpy()) for smi in smiles]
            next_output_data["reward"][done] = torch.tensor(
                scoring_function(smiles_str), device=output_data.device
            ).unsqueeze(-1)

        return output_data


def _get_log_prob(
    smiles: list,
    policy: Union[TensorDictModule, Callable[[TensorDictBase], TensorDictBase]],
    vocabulary: Vocabulary,
    max_length: int,
    sum_log_prob: bool = True,
):
    # Convert SMILES to TensorDict
    tokens = [torch.tensor(vocabulary.encode(smi)) for smi in smiles]
    enc_smiles = torch.vstack(
        [
            # Pad with -1 in case the end token is already 0
            torch.nn.functional.pad(tok, (0, max_length - tok.size()[0]), value=-1)
            for tok in tokens
        ]
    )
    data = smiles_to_tensordict(
        enc_smiles, mask_value=-1, replace_mask_value=0, device=policy.device
    )
    actions = data.get("action")

    # For transformers-based policies
    data.set("sequence", data.get("observation"))
    data.set("sequence_mask", data.get("mask"))

    log_prob = policy.get_dist(data).log_prob(actions)
    if sum_log_prob:
        log_prob = log_prob.sum(-1)
    return log_prob.cpu()
