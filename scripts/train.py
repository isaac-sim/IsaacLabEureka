# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: MIT

"""Script to train an RL agent with Isaac Lab Eureka."""

import argparse

from isaaclab_eureka.eureka import Eureka


def main(args_cli):
    eureka = Eureka(
        task=args_cli.task,
        rl_library=args_cli.rl_library,
        num_parallel_runs=args_cli.num_parallel_runs,
        device=args_cli.device,
        max_training_iterations=args_cli.max_training_iterations,
        feedback_subsampling=args_cli.feedback_subsampling,
        temperature=args_cli.temperature,
        gpt_model=args_cli.gpt_model,
    )

    eureka.run(max_eureka_iterations=args_cli.max_eureka_iterations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with Eureka.")
    parser.add_argument("--task", type=str, default="Isaac-Cartpole-Direct-v0", help="Name of the task.")
    parser.add_argument(
        "--num_parallel_runs", type=int, default=1, help="Number of Eureka runs to execute in parallel."
    )
    parser.add_argument("--device", type=str, default="cuda", help="The device to run training on.")
    parser.add_argument("--max_eureka_iterations", type=int, default=5, help="The number of Eureka iterations to run.")
    parser.add_argument(
        "--max_training_iterations",
        type=int,
        default=100,
        help="The number of RL training iterations to run for each Eureka iteration.",
    )
    parser.add_argument(
        "--feedback_subsampling",
        type=int,
        default=10,
        help="The subsampling of the metrics given as feedack to the LLM.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Controls the randomness of the GPT output (0 is deterministic, 1 is highly diverse).",
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-4", help="The GPT model to use.")
    parser.add_argument("--rl_library", type=str, default="rsl_rl", help="The RL training library to use.")
    args_cli = parser.parse_args()

    # Run the main function
    main(args_cli)
