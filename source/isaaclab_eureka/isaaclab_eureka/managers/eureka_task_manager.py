# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import math
import multiprocessing
import os
import traceback
import types
from contextlib import nullcontext
from datetime import datetime
from typing import Literal

from isaaclab_eureka.utils import MuteOutput, get_freest_gpu

TEMPLATE_REWARD_STRING = """
from {module_name} import *
import torch

def _get_rewards(self):
    rewards_oracle = self._get_rewards_oracle()
    rewards_eureka, rewards_dict = self._get_rewards_eureka()
    self._eureka_episode_sums["eureka_total_rewards"] += rewards_eureka
    self._eureka_episode_sums["oracle_total_rewards"] += rewards_oracle
    for key in rewards_dict.keys():
        if key not in self._eureka_episode_sums:
            self._eureka_episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
        self._eureka_episode_sums[key] += rewards_dict[key]
    return rewards_eureka
"""


# Insert the logic to log the eureka episode sums.
TEMPLATE_RESET_STRING = """
from {module_name} import *

@torch.inference_mode()
def _reset_idx(self, env_ids):
    if env_ids is None or len(env_ids) == self.num_envs:
        env_ids = torch.arange(self.num_envs, device=self.device)
    extras = dict()
    # This needs to happen before self._reset_idx_original(env_ids) because it will reset buffers that might be needed
    {success_metric}
    self._reset_idx_original(env_ids)
    if not "log" in self.extras:
        self.extras["log"] = dict()
    for key in self._eureka_episode_sums.keys():
        episodic_sum_avg = torch.mean(self._eureka_episode_sums[key][env_ids])
        extras["Eureka/"+key] = episodic_sum_avg / self.max_episode_length_s
        self._eureka_episode_sums[key][env_ids] = 0.0
    self.extras["log"].update(extras)
"""


class EurekaTaskManager:
    """Manages the set-up and training of a task using LLM-generated reward functions.

    It takes an existing IsaacLab task and inserts the Eureka-generated reward function or configuration into it. The
    rewards that are already defined in the task are kept to serve as an oracle signal.
    """

    def __init__(
        self,
        task: str,
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        num_processes: int = 1,
        device: str = "cuda",
        env_seed: int = 42,
        max_training_iterations: int = 100,
        success_metric_string: str = "",
    ):
        """Initialize the task manager. Each process will create an independent training run.

        Args:
            task: The name of the task to train.
            rl_library: The RL library to use for training.
            num_processes: The number of processes to use for training.
            device: The device to run training on.
            env_seed: The seed to use for the environment.
            max_training_iterations: The maximum number of training iterations.
            success_metric_string: A string that represents an expression to calculate the success metric for the task.
        """
        self._task = task
        self._rl_library = rl_library
        self._num_processes = num_processes
        self._device = device
        self._max_training_iterations = max_training_iterations
        self._success_metric_string = success_metric_string
        self._env_seed = env_seed
        if self._success_metric_string:
            self._success_metric_string = "extras['Eureka/success_metric'] = " + self._success_metric_string

        self._processes = dict()
        # Used to communicate the reward functions to the processes
        self._rewards_queues = [multiprocessing.Queue() for _ in range(self._num_processes)]
        # Used to communicate the observations method to the main process
        self._observations_queue = multiprocessing.Queue()
        # Used to communicate the results of the training runs to the main process
        self._results_queue = multiprocessing.Queue()
        # Used to signal the processes to terminate
        self.termination_event = multiprocessing.Event()

        for idx in range(self._num_processes):
            p = multiprocessing.Process(target=self._worker, args=(idx, self._rewards_queues[idx]))
            self._processes[idx] = p
            p.start()

        # Fetch the observations
        self._get_observations_as_string = self._observations_queue.get()

    @property
    def get_observations_method_as_string(self) -> str:
        """The _get_observations method of the environment as a string."""
        return self._get_observations_as_string

    def close(self):
        """Close the task manager and clean up the processes."""
        self.termination_event.set()
        # Send a stop signal to the processes
        for rewards_queue in self._rewards_queues:
            rewards_queue.put("Stop")
        for process in self._processes.values():
            process.join()

    def train(self, get_rewards_method_as_string: list[str]) -> list[dict]:
        """Train the task with the specified reward functions.

        Note: The methods must have the following signature "_get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]".

        Args:
            get_rewards_method_as_string: A list of get_rewards methods as strings. The length of the list must match
            the number of processes.
        Returns:
            A list of dictionaries containing the results of each training run. The dictionary contains the following
            keys:
                - "success": True if the training was successful, False otherwise.
                - "log_dir": The directory where the training logs are stored if the training succeeded.
                - "exception": The exception message if the training failed.
        """
        if len(get_rewards_method_as_string) != self._num_processes:
            raise ValueError(
                f"Number of reward methods in the list ({len(get_rewards_method_as_string)}) does not match the number"
                f" of processes ({self._num_processes})."
            )

        # Set the reward functions in each process
        for idx, rewards_queue in enumerate(self._rewards_queues):
            rewards_queue.put(get_rewards_method_as_string[idx])

        results = [None] * self._num_processes
        # Wait for each process to finish and collect the results
        for _ in range(self._num_processes):
            idx, result = self._results_queue.get()
            results[idx] = result

        return results

    def _worker(self, idx: int, rewards_queue: multiprocessing.Queue):
        """The worker function that runs the training of the task.

        Args:
            idx: The index of the worker.
            rewards_queue: The queue to receive the reward function from the main process
        """
        self._idx = idx
        while not self.termination_event.is_set():
            if not hasattr(self, "_env"):
                self._create_environment()

                # Fetch the environment's _get_observations method and send it to the main process
                if self._idx == 0 and not hasattr(self, "_observation_string"):
                    self._observation_string = inspect.getsource(self._env.unwrapped._get_observations)
                    self._observations_queue.put(self._observation_string)

            # Insert the reward function into the environment and run the training
            reward_func_string = rewards_queue.get()
            if isinstance(reward_func_string, str) and reward_func_string.startswith("def _get_rewards_eureka(self)"):
                try:
                    self._prepare_eureka_environment(reward_func_string)
                    # Only print the output of process 0
                    context = MuteOutput() if self._idx > 0 else nullcontext()
                    with context:
                        # Run training and send result to main process
                        self._run_training()
                    result = {"success": True, "log_dir": self._log_dir}
                except Exception as e:
                    result = {"success": False, "exception": str(e)}
                    print(traceback.format_exc())
            else:
                result = {
                    "success": False,
                    "exception": (
                        "The reward function must be a string that starts with 'def _get_rewards_eureka(self)'."
                    ),
                }

            self._results_queue.put((self._idx, result))
        # Clean up
        print(f"[INFO]: Run {self._idx} terminated.")
        self._env.close()
        self._simulation_app.close()

    def _create_environment(self):
        """Create the environment for the task."""
        from isaaclab.app import AppLauncher

        if self._device == "cuda":
            device_id = get_freest_gpu()
            self._device = f"cuda:{device_id}"
        app_launcher = AppLauncher(headless=True, device=self._device)
        self._simulation_app = app_launcher.app

        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import DirectRLEnvCfg
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg: DirectRLEnvCfg = parse_env_cfg(self._task)
        env_cfg.sim.device = self._device
        env_cfg.seed = self._env_seed
        self._env = gym.make(self._task, cfg=env_cfg)

    def _prepare_eureka_environment(self, get_rewards_method_as_string: str):
        """Prepare the environment for training with the Eureka-generated reward function.

        It renames the original reward function to _get_rewards_oracle, adds the Eureka-generated reward function to the
        environment, and sets the environment's _get_rewards method to a template method that calls both the Eureka and
        oracle reward functions. It also sets the environment's _reset_idx method to a template method that updates the
        episodic sum of the Eureka-generated rewards.
        """
        import torch

        env = self._env.unwrapped
        namespace = {}
        # Check if the environment has already been prepared
        if not hasattr(env, "_get_rewards_eureka"):
            # rename the environment's original reward function to _get_rewards_oracle
            env._get_rewards_oracle = env._get_rewards
            # rename to environment's initial reset function to _reset_idx_original
            env._reset_idx_original = env._reset_idx
            # set the _get_rewards method to the template method
            template_reward_string_with_module = TEMPLATE_REWARD_STRING.format(module_name=env.__module__)
            exec(template_reward_string_with_module, namespace)
            setattr(env, "_get_rewards", types.MethodType(namespace["_get_rewards"], env))
            # set the _reset_idx method to the template method
            template_reset_string_with_success_metric = TEMPLATE_RESET_STRING.format(
                module_name=env.__module__, success_metric=self._success_metric_string
            )
            # hack: can't enable inference with rl_games
            if self._rl_library == "rl_games":
                template_reset_string_with_success_metric = template_reset_string_with_success_metric.replace(
                    "@torch.inference_mode()", ""
                )
            exec(template_reset_string_with_success_metric, namespace)
            setattr(env, "_reset_idx", types.MethodType(namespace["_reset_idx"], env))

        # Add the GPT generated reward function to the environment
        get_rewards_method_as_string = f"from {env.__module__} import * \nimport torch\n" + get_rewards_method_as_string
        exec(get_rewards_method_as_string, namespace)
        setattr(env, "_get_rewards_eureka", types.MethodType(namespace["_get_rewards_eureka"], env))

        # Prepare the reward sum buffers
        env._eureka_episode_sums = dict()
        env._eureka_episode_sums["eureka_total_rewards"] = torch.zeros(env.num_envs, device=env.device)
        env._eureka_episode_sums["oracle_total_rewards"] = torch.zeros(env.num_envs, device=env.device)

    def _run_training(self, framework: Literal["rsl_rl", "rl_games"] = "rsl_rl"):
        """Run the training of the task."""
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        if self._rl_library == "rsl_rl":
            from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(self._task, "rsl_rl_cfg_entry_point")
            agent_cfg.device = self._device
            agent_cfg.max_iterations = self._max_training_iterations

            log_root_path = os.path.join("logs", "rl_runs", "rsl_rl_eureka", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs: {time-stamp}_{run_name}
            log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_Run-{self._idx}"
            if agent_cfg.run_name:
                log_dir += f"_{agent_cfg.run_name}"
            self._log_dir = os.path.join(log_root_path, log_dir)

            env = RslRlVecEnvWrapper(self._env)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=self._log_dir, device=agent_cfg.device)
            runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        elif self._rl_library == "rl_games":
            from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
            from rl_games.common import env_configurations, vecenv
            from rl_games.common.algo_observer import IsaacAlgoObserver
            from rl_games.torch_runner import Runner

            agent_cfg = load_cfg_from_registry(self._task, "rl_games_cfg_entry_point")
            agent_cfg["params"]["config"]["max_epochs"] = self._max_training_iterations
            agent_cfg["params"]["config"]["device"] = self._device
            agent_cfg["params"]["config"]["device_name"] = self._device
            # specify directory for logging experiments
            log_root_path = os.path.join("logs", "rl_runs", "rl_games_eureka", agent_cfg["params"]["config"]["name"])
            log_root_path = os.path.abspath(log_root_path)
            print(f"[INFO] Logging experiment in directory: {log_root_path}")
            # specify directory for logging runs
            log_dir = (
                agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
                + f"_Run-{self._idx}"
            )
            # set directory into agent config
            # logging directory path: <train_dir>/<full_experiment_name>
            agent_cfg["params"]["config"]["train_dir"] = log_root_path
            agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
            # Update the log directory to the tensorboard file
            self._log_dir = os.path.join(log_root_path, log_dir, "summaries")
            clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
            clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
            env = RlGamesVecEnvWrapper(self._env, self._device, clip_obs, clip_actions)

            vecenv.register(
                "IsaacRlgWrapper",
                lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
            )
            env_configurations.register(
                "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
            )

            # set number of actors into agent config
            agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
            # create runner from rl-games
            runner = Runner(IsaacAlgoObserver())
            runner.load(agent_cfg)
            # reset the agent and env
            runner.reset()
            # train the agent
            runner.run({"train": True, "play": False, "sigma": None})
        else:
            raise Exception(f"framework {framework} is not supported yet.")
