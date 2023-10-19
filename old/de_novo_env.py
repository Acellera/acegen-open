import gymnasium as gym


class DeNovoEnv(gym.Env):
    """Custom Environment for Generative Chemistry RL."""

    def __init__(
        self,
        start_token: int,
        end_token: int,
        length_vocabulary: int,
        max_length: int = 100,
    ) -> None:
        self.start_token = int(start_token)
        self.end_token = int(end_token)
        self.max_length = max_length

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(length_vocabulary)
        self.observation_space = gym.spaces.Discrete(length_vocabulary)

    def step(self, action):
        """Execute one time step within the rl_environments"""

        # Get next action
        self.current_episode_length += 1

        action = int(
            self.end_token
            if self.current_episode_length
            == self.max_length - 2  # account for start and end tokens
            else action
        )

        reward = 0.0
        done = False
        info = {}

        # Handle end of molecule/episode if action is $
        if action == self.end_token:
            done = True

        # Define next observation
        next_obs = action
        truncated = False
        return next_obs, reward, done, truncated, info

    def reset(self):
        """
        Reset the state of the rl_environments to an initial state.
        Return padded base molecule to match length `obs_length`.
        """
        self.current_episode_length = 1
        obs = self.start_token
        info = {}

        return obs, info

