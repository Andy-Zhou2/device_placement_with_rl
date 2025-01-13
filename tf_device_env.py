import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, Unbounded
from tf_device_measure import measure_inference_time_3devices
from torch import Size
from torchrl.envs.utils import check_env_specs, step_mdp

class DevicePlacementEnv(EnvBase):
    def __init__(self, device='cpu'):
        super().__init__(device=device)
        self.action_spec = Bounded(shape=Size([3]), low=0, high=2, dtype=torch.int64, device=device)
        self.reward_spec = Bounded(shape=Size([1]), low=float('-inf'), high=0, dtype=torch.float32, device=device)
        self.observation_spec = Composite({
        })
        self.episode_done = False

    def _reset(self, tensordict):
        self.episode_done = False
        return TensorDict({}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict.get("action")
        reward = measure_inference_time_3devices(action) * (-1)
        self.episode_done = True
        next_tensordict = TensorDict({
            "reward": reward,
            "done": torch.tensor([self.episode_done], dtype=torch.bool, device=self.device)
        }, batch_size=[])
        return next_tensordict

    def _set_seed(self, seed):
        torch.manual_seed(seed)

if __name__ == "__main__":
    env = DevicePlacementEnv()
    # check_env_specs(env)
    print(env.reset())
    env.step(TensorDict({"action": torch.tensor([0, 1, 2], dtype=torch.int64)}))