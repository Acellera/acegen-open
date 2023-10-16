import numpy as np
from rdkit import Chem
import gymnasium as gym


class DeNovoEnv(gym.Env):
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

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Discrete(len(self.vocabulary))

    def step(self, action):
        """Execute one time step within the rl_environments"""

        # Get next action
        self.current_episode_length += 1

        action = (
            "$"
            if self.current_episode_length
            == self.max_length - 2  # account for start and end tokens
            else self.vocabulary.decode_token(action)
        )

        # Update current SMILES
        self.current_molecule_str += action

        reward = 0.0
        done = False
        info = {}

        # Handle end of molecule/episode if action is $
        if action == "$":
            # Set done flag
            done = True

            # # Get SMILES
            # smiles = self.vocabulary.remove_start_and_end_tokens(
            #     self.current_molecule_str
            # )
            # info["molecule"] = smiles
            #
            # # check smiles validity
            # mol = Chem.MolFromSmiles(smiles)
            #
            # if mol is not None:
            #     # Compute score
            #     # score = self.scoring_function([smiles], flt=True)
            #
            #     # Get reward or score
            #     reward = 0.0

        # Define next observation
        next_obs = self.vocabulary.encode_token(action).astype(np.int64)
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        """
        Reset the state of the rl_environments to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_molecule_str = "^"
        self.current_episode_length = 1
        obs = self.vocabulary.encode_token("^").astype(np.int64)
        info = {}

        return obs, info

