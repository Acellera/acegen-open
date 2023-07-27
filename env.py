import gym
import numpy as np
from rdkit import Chem


class GenChemEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    def __init__(
        self,
        scoring_function,
        vocabulary,
        max_length=100,
    ):

        self.max_length = max_length
        self.vocabulary = vocabulary
        self.scoring_function = scoring_function

        # Scoring example
        test_smiles = "C"
        self.scoring_example = scoring_function(test_smiles)
        self.scoring_example.update({"molecule": test_smiles, "reaction_scores": 0.0, "repeated": 0.0})

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        # self.observation_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Box(low=0, high=len(self.vocabulary) - 1, shape=(1, ), dtype=np.int64)

    def step(self, action):
        """Execute one time step within the environment"""

        # Get next action
        action = (
            "$"
            if self.current_episode_length == self.max_length - 1
            else self.vocabulary.decode_token(action)
        )

        # Update current SMILES
        self.current_molecule_str += action
        self.current_episode_length += 1

        reward = 0.0
        done = False
        info = {}

        # Handle end of molecule/episode if action is $
        if action == "$":

            # Set done flag
            done = True

            # Get smile
            smiles = self.vocabulary.remove_start_and_end_tokens(self.current_molecule_str)

            # check smiles validity
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:

                # Compute score
                score = self.scoring_function(smiles)

                # Get reward or score
                reward = score["reward"]

        # Define next observation
        next_obs = self.vocabulary.encode_token(action)
        return next_obs, reward, done, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_molecule_str = "^"
        self.current_episode_length = 1
        obs = self.vocabulary.encode_token("^")
        info = {k: 0.0 for k, v in self.scoring_example.items()}

        return obs, info
