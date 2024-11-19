import numpy as np
from collections import deque


class PerStepPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        num_steps = rewards.shape[-1]
        unique_prompts = np.unique(prompts)
        advantages = np.empty_like(rewards)

        for prompt in unique_prompts:
            for step in range(num_steps):
                prompt_step_rewards = rewards[prompts == prompt, step]
                if prompt not in self.stats:
                    self.stats[prompt] = {}
                if step not in self.stats[prompt]:
                    self.stats[prompt][step] = deque(maxlen=self.buffer_size)
                self.stats[prompt][step].extend(prompt_step_rewards)

                if len(self.stats[prompt][step]) < self.min_count:
                    mean = np.mean(rewards[:, step])
                    std = np.std(rewards[:, step]) + 1e-6
                else:
                    mean = np.mean(self.stats[prompt][step])
                    std = np.std(self.stats[prompt][step]) + 1e-6

                advantages[prompts == prompt, step] = (prompt_step_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }
