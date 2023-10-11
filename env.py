import os
import csv
import gym
import time
import json
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
        self.observation_space = gym.spaces.Discrete(len(self.vocabulary))

    def step(self, action):
        """Execute one time step within the environment"""

        # Get next action
        self.current_episode_length += 1

        print(self.current_episode_length)

        action = (
            "$"
            if self.current_episode_length == self.max_length - 2  # account for start and end tokens
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

            # Get smile
            smiles = self.vocabulary.remove_start_and_end_tokens(self.current_molecule_str)
            info["molecule"] = smiles

            # check smiles validity
            mol = Chem.MolFromSmiles(smiles)

            if mol is not None:

                # Compute score
                score = self.scoring_function(smiles)

                # Get reward or score
                reward = score["reward"]

        # Define next observation
        next_obs = self.vocabulary.encode_token(action).astype(np.int64)
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_molecule_str = "^"
        self.current_episode_length = 1
        obs = self.vocabulary.encode_token("^").astype(np.int64)
        info = {k: 0.0 for k, v in self.scoring_example.items()}

        return obs, info


class ResultsWriter:
    def __init__(self, filename, header, extra_keys=()):
        self.extra_keys = extra_keys
        already_exists = os.path.isfile(filename)
        self.f = open(filename, "a+")
        self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
        if not already_exists:
            header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.f.flush()
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


class Monitor(gym.Wrapper):

    def __init__(self, env, log_dir, info_keywords=("molecule", )):
        super(Monitor, self).__init__(env)
        self.f = None
        self.tstart = time.time()
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"monitor_{os.getpid()}_{id(self)}.csv")
        self.results_writer = ResultsWriter(
            filename,
            header={"t_start": time.time()},
            extra_keys=info_keywords)
        self.info_keywords = info_keywords
        self.rewards = None

    def reset(self, **kwargs):
        self.reset_state()
        return self.env.reset(**kwargs)

    def reset_state(self):
        self.rewards = []

    def step(self, action):
        ob, rew, done, truncated, info = self.env.step(action)
        self.update(ob, rew, done, info)
        truncated = False
        return ob, rew, done, truncated, info

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            eprew = 0.0 + sum(self.rewards)
            eplen = 1.0 + len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.results_writer.write_row(epinfo)

    def close(self):
        super(Monitor, self).close()
        if self.f is not None:
            self.f.close()

