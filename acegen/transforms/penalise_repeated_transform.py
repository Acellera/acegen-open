import torch
from tensordict import TensorDictBase
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.transforms.transforms import Transform


class PenaliseRepeatedSMILES(Transform):
    def __init__(
        self,
        check_duplicate_key,
        in_key=None,
        out_key=None,
        penalty=0.0,
        device=None,
        max_tracked_smiles=10_000,
    ):
        """Penalise repeated smiles and add unique smiles to the diversity buffer.

        Args:
            check_duplicate_key: The key in the tensordict that contains the smiles.
            in_key: The key in the tensordict that contains the reward.
            out_key: The key in the tensordict to store the penalised reward.
            penalty: The penalty to apply to the reward.
            device: The device to store the diversity buffer on.
            max_tracked_smiles: number of SMILES to track for repetition checking..
        """
        self.penalty = penalty
        self.check_duplicate_key = check_duplicate_key
        self._repeated_smiles = 0
        self.diversity_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_tracked_smiles, device=device),
        )

        if in_key is None:
            in_key = ["reward"]
        if out_key is None:
            out_key = ["reward"]

        super().__init__(in_key, out_key)

    @property
    def repeated_smiles(self):
        return self._repeated_smiles

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Penalise repeated smiles and add unique smiles to the diversity buffer."""
        # Check duplicated key is in tensordict
        if self.check_duplicate_key not in tensordict.keys():
            raise KeyError(
                f"duplicate_key {self.check_duplicate_key} not found in tensordict."
            )

        # Get a td with only the terminated trajectories
        td_next = tensordict.get("next")
        terminated = td_next.get("terminated").squeeze(-1)
        sub_td = td_next.get_sub_tensordict(idx=terminated)

        # Get current smiles and reward
        finished_smiles = sub_td.get(self.check_duplicate_key)
        reward = sub_td.get(*self.in_keys)

        # Get smiles found so far
        td_smiles = self.diversity_buffer._storage._storage

        # Identify repeated smiles
        repeated = torch.zeros(
            finished_smiles.shape[0], dtype=torch.bool, device=tensordict.device
        )
        if td_smiles is not None:
            unique_smiles = td_smiles.get(("_data", self.check_duplicate_key))[
                : len(self.diversity_buffer)
            ]
            for i, smi in enumerate(finished_smiles):
                repeated[i] = (smi == unique_smiles).all(dim=-1).any()

        # Apply penalty
        repeated = repeated & (reward > 0).squeeze()
        reward[repeated] = reward[repeated] * self.penalty
        sub_td.set(*self.out_keys, reward, inplace=True)

        # Add unique smiles to the diversity buffer
        if (~repeated).any():
            self.diversity_buffer.extend(
                sub_td.select(self.check_duplicate_key)[~repeated]
            )

        self._repeated_smiles += repeated.sum().item()

        return tensordict
