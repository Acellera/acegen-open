import os
import csv
import time
import json
import gymnasium as gym


class ResultsWriter:
    def __init__(self, filename, header, extra_keys=()):
        self.extra_keys = extra_keys
        already_exists = os.path.isfile(filename)
        self.f = open(filename, "a+")
        self.logger = csv.DictWriter(
            self.f, fieldnames=("r", "l", "t") + tuple(extra_keys)
        )
        if not already_exists:
            header = "# {} \n".format(json.dumps(header))
            self.f.write(header)
            self.f.flush()
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


class Monitor(gym.Wrapper):
    def __init__(self, env, log_dir, info_keywords=("molecule",)):
        super(Monitor, self).__init__(env)
        self.f = None
        self.tstart = time.time()
        os.makedirs(log_dir, exist_ok=True)
        filename = os.path.join(log_dir, f"monitor_{os.getpid()}_{id(self)}.csv")
        self.results_writer = ResultsWriter(
            filename, header={"t_start": time.time()}, extra_keys=info_keywords
        )
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
            epinfo = {
                "r": eprew,
                "l": eplen,
                "t": time.time() - self.tstart,
            }
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.results_writer.write_row(epinfo)

    def close(self):
        super(Monitor, self).close()
        if self.f is not None:
            self.f.close()