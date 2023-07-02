import gym
from rdkit import Chem
from collections import deque


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    def __init__(
        self,
        scoring_function,
        vocabulary,
        max_length=60,
    ):

        super(GenChemEnv, self).__init__(
            scoring_function,
            vocabulary,
            max_length,
        )

        self.max_length = max_length
        self.vocabulary = vocabulary
        self.scoring_function = scoring_function
        self.running_mean_valid_smiles = deque(maxlen=100)

        # Scoring example
        test_smiles = "C"
        self.scoring_example = scoring_function(test_smiles)
        self.scoring_example.update({"molecule": test_smiles, "reaction_scores": 0.0, "repeated": 0.0})

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        observation_space = gym.spaces.Discrete(len(self.vocabulary))
        observation_length = gym.spaces.Discrete(self.max_length)
        self.observation_space = gym.spaces.Dict(
            {
                "obs": observation_space,
                "obs_length": observation_length,
            }
        )

    def step(self, action):
        """Execute one time step within the environment"""

        info = {k: 0.0 for k, v in self.scoring_example.items()}
        info.update(
            {
                "molecule": "invalid",
                "valid_smile": False,
                "repeated": 0.0,
            }
        )

        # Get next action
        action = (
            "$"
            if self.current_episode_length == self.max_length - 1
            else self.vocabulary.decode_token(action)
        )

        # Update current SMILES
        self.current_molecule_str += action
        self.current_episode_length += 1

        # Handle end of molecule/episode if action is $
        reward = 0.0
        done = False
        if action == "$":

            # Get smile
            smiles = self.vocabulary.remove_start_and_end_tokens(
                self.current_molecule_str
            )

            # check smiles validity
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:

                # Compute score
                score = self.scoring_function(smiles)

                # Sanity check
                if not (isinstance(score, dict) and "reward" in score.keys()):
                    raise ValueError(
                        "scoring_function has to return a dict with at least the keyword ´reward´"
                    )

                # Get reward or score
                reward = score["reward"]

                # Update info and done flag
                info.update(score)
                self.running_mean_valid_smiles.append(True)
                info.update({"valid_smile": True})

            else:
                self.running_mean_valid_smiles.append(False)

            info.update(
                {
                    "running_mean_valid_smiles": float(
                        (
                            sum(self.running_mean_valid_smiles)
                            / len(self.running_mean_valid_smiles)
                        )
                        * 100
                    ),
                    "molecule": smiles,
                }
            )

            # Set done flag
            done = True

        # Define next observation
        next_obs = {
            "obs": self.vocabulary.encode_token(action),
            "obs_length": 1,
        }

        return next_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_molecule_str = "^"
        self.current_episode_length = 1
        obs = {
            "obs":  self.vocabulary.encode_token("^"),
            "obs_length": 1,
        }
        info = {k: 0.0 for k, v in self.scoring_example.items()}

        return obs, info
