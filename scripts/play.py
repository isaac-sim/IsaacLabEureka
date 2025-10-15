# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play an RL agent with Isaac Lab Eureka."""

import argparse
import math

from isaaclab_eureka.utils import get_freest_gpu


def main(args_cli):
    """Create the environment for the task."""
    from isaaclab.app import AppLauncher

    # parse args from cmdline
    device = args_cli.device
    task = args_cli.task
    checkpoint = args_cli.checkpoint

    # parse device
    if device == "cuda":
        device_id = get_freest_gpu()
        device = f"cuda:{device_id}"

    # launch app
    app_launcher = AppLauncher(headless=args_cli.headless, device=device)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg: DirectRLEnvCfg = parse_env_cfg(task)
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env = gym.make(task, cfg=env_cfg)

    """Run the inferencing of the task."""
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    if args_cli.rl_library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
        agent_cfg.device = device

        env = RslRlVecEnvWrapper(env)
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(checkpoint)
        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        # reset environment
        obs = env.get_observations()
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)

    elif args_cli.rl_library == "rl_games":
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from rl_games.common import env_configurations, vecenv
        from rl_games.common.algo_observer import IsaacAlgoObserver
        from rl_games.torch_runner import Runner

        agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
        # parse checkpoint path
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = checkpoint
        agent_cfg["params"]["config"]["device"] = device
        agent_cfg["params"]["config"]["device_name"] = device
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
        env = RlGamesVecEnvWrapper(env, device, clip_obs, clip_actions)

        vecenv.register(
            "IsaacRlgWrapper",
            lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
        )
        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        # create runner from rl-games
        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)

        # obtain the agent from the runner
        # we import this here to avoid GLIBC errors with Isaac Sim 5.0 in conda
        from rl_games.common.player import BasePlayer

        agent: BasePlayer = runner.create_player()
        agent.restore(checkpoint)
        agent.reset()

        # reset environment
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        # required: enables the flag for batched observations
        _ = agent.get_batch_size(obs, 1)
        # initialize RNN states if used
        if agent.is_rnn:
            agent.init_rnn()
        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=True)
                # env stepping
                obs, _, dones, _ = env.step(actions)

                # perform operations for terminated episodes
                if len(dones) > 0:
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, dones, :] = 0.0

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Absolute path to model checkpoint.")
    parser.add_argument(
        "--rl_library",
        type=str,
        default="rsl_rl",
        choices=["rsl_rl", "rl_games"],
        help="The RL training library to use.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Force display off at all times.",
    )
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)
