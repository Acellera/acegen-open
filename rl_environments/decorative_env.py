import numpy as np
import gymnasium as gym
from reinvent_chemistry import Conversions
from reinvent_chemistry.library_design import BondMaker, AttachmentPoints


class DecorativeEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    def __init__(
        self,
        vocabulary,
        scaffold,
        max_length=100,
    ):
        self.scaffold = scaffold
        self.encoded_scaffold = vocabulary.encode_scaffold(scaffold)
        self.max_length = max_length
        self.vocabulary = vocabulary

        self._bond_maker = BondMaker()
        self._conversion = Conversions()
        self._attachment_points = AttachmentPoints()

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))
        self.observation_space = gym.spaces.Discrete(len(self.vocabulary.decoration_vocabulary))

        # Start and end tokens
        self.start_token = self.vocabulary.encode_token("^").astype(np.int64)
        self.end_token = self.vocabulary.encode_token("$").astype(np.int64)

    def step(self, action):
        """Execute one time step within the rl_environments"""

        # Get next action
        self.current_episode_length += 1

        action = (
            self.end_token
            if self.current_episode_length
               == self.max_length - 2  # account for start and end tokens
            else action
        )

        reward = 0.0
        done = False
        info = {"scaffold": self.scaffold}

        # Handle end of molecule/episode if action is $
        if action == self.end_token:
            # Set done flag
            done = True

            # Get SMILES
            smiles, mol = self.join_scaffold_and_decorations(
                self.scaffold,
                self.vocabulary.remove_start_and_end_tokens(self.current_molecule_str),
            )
            info["molecule"] = smiles

        # Define next observation
        next_obs = self.action_space
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        """
        Reset the state of the rl_environments to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_episode_length = 1
        obs = self.start_token
        info = {"scaffold": self.scaffold}

        return obs, info

    def join_scaffold_and_decorations(self, scaffold, decorations):
        scaffold = self._attachment_points.add_attachment_point_numbers(
            scaffold, canonicalize=False
        )
        molecule = self._bond_maker.join_scaffolds_and_decorations(
            scaffold, decorations
        )
        smile = self._conversion.mol_to_smiles(molecule) if molecule else None
        return smile, molecule
