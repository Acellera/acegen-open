from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer
from torchrl.envs.transforms.transforms import Transform


class PenaliseRepeatedSMILES(Transform):
    def __init__(
            self,
            diversity_buffer,
            duplicate_key,
            in_key=None,
            out_key=None,
            penalty=0.0,
    ):
        """Penalise repeated smiles and add unique smiles to the diversity buffer.

        Args:
            diversity_buffer: A TensorDictReplayBuffer instance.
            duplicate_key: The key in the tensordict that contains the smiles.
            in_key: The key in the tensordict that contains the reward.
            out_key: The key in the tensordict to store the penalised reward.
            penalty: The penalty to apply to the reward.
        """
        self.penalty = penalty
        self.duplicate_key = duplicate_key
        self.diversity_buffer = diversity_buffer
        self._repeated_smiles = 0

        if not isinstance(diversity_buffer, TensorDictReplayBuffer):
            raise ValueError("diversity_buffer must be a TensorDictReplayBuffer instance.")

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
        if self.duplicate_key not in tensordict.keys():
            raise KeyError(f"duplicate_key {self.duplicate_key} not found in tensordict.")

        td_next = tensordict.get("next")
        terminated = td_next.get("terminated").squeeze(-1)
        sub_td = td_next.get_sub_tensordict(idx=terminated)

        reward = sub_td.get(*self.in_keys)
        num_unique_smiles = len(self.diversity_buffer)
        finished_smiles = sub_td.get(self.duplicate_key)
        finished_smiles_td = sub_td.select(self.duplicate_key)
        num_finished_smiles = len(finished_smiles_td)

        if num_finished_smiles > 0 and num_unique_smiles == 0:
            self.diversity_buffer.extend(finished_smiles_td)

        elif num_finished_smiles > 0:
            for i, smi in enumerate(finished_smiles):
                td_smiles = self.diversity_buffer._storage._storage
                unique_smiles = td_smiles.get(("_data", self.duplicate_key))[:num_unique_smiles]
                repeated = (smi == unique_smiles).all(dim=-1).any()
                import ipdb; ipdb.set_trace()
                if repeated:
                    reward[i] = reward[i] * self.penalty
                    self._repeated_smiles += 1
                elif reward[i] > 0:
                    self.diversity_buffer.extend(finished_smiles_td[i: i + 1])
                    num_unique_smiles += 1

        sub_td.set(*self.out_keys, reward, inplace=True)
        return tensordict
