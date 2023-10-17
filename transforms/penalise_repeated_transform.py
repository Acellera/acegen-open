from typing import Callable
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform
from vocabulary.vocabulary import DeNovoVocabulary


class PenaliseRepeated(Transform):
    def __init__(
            self,
            reward_function: Callable,
            vocabulary: DeNovoVocabulary,
            in_keys=None,
            out_keys=None,
            on_done_only=True,
            truncated_key="truncated",
            use_next: bool = True,
            gradient_mode=False,
    ):
        self.on_done_only = on_done_only
        self.truncated_key = truncated_key
        self.use_next = use_next
        self.gradient_mode = gradient_mode

        if not isinstance(reward_function, Callable):
            raise ValueError("reward_function must be a callable.")

        if out_keys is None:
            out_keys = ["reward"]
        if in_keys is None:
            in_keys = ["SMILES"]

        super().__init__(in_keys, out_keys)

        self.vocabulary = vocabulary
        self.reward_function = reward_function

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        pass
