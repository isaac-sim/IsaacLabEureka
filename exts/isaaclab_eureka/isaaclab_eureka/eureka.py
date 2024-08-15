# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter
from typing import Literal

from isaaclab_eureka import EUREKA_ROOT_DIR
from isaaclab_eureka.config import (
    DIRECT_WORKFLOW_INITIAL_PROMPT,
    DIRECT_WORKFLOW_TASK_PROMT,
    TASK_FAILURE_FEEBACK_PROMPT,
    TASK_SUCCESS_POST_FEEDBACK_PROMT,
    TASK_SUCCESS_PRE_FEEDBACK_PROMT,
    TASKS_CFG,
)
from isaaclab_eureka.managers import EurekaTaskManager, LLMManager
from isaaclab_eureka.utils import load_tensorboard_logs


class Eureka:
    """Orchestrates the training of the RL agent using the LLM."""

    def __init__(
        self,
        task: str,
        device: str = "cuda",
        rl_library: Literal["rsl_rl", "rl_games"] = "rsl_rl",
        max_training_iterations: int = 100,
        feedback_subsampling: int = 10,
        temperature: float = 1.0,
        gpt_model: str = "gpt-4",
        num_parallel_runs: int = 1,
    ):
        """Initialize the Eureka class.

        Args:

            task: The task to train the agent on.
            device: The device to run the training on.
            rl_library: The RL library to use for training.
            max_training_iterations: The maximum number of training iterations for the RL agent.
            feedback_subsampling: The subsampling of the metrics given as feedack to the LLM.
            temperature: The temperature to use for the GPT model.
            gpt_model: The GPT model to use.
            num_parallel_runs: The number of runs to execute in parallel.
        """

        # Load the task description and success metric
        if task in TASKS_CFG:
            task_description = TASKS_CFG[task]["description"]
            successs_metric_string = TASKS_CFG[task].get("successs_metric")
            self._successs_metric_to_win = TASKS_CFG[task].get("successs_metric_to_win")
        else:
            raise ValueError(
                f"Task configuration for {task} not found in the `TASKS_CFG` dictionary in config/tasks.py."
            )

        self._task_description = task_description
        self._feedback_subsampling = feedback_subsampling
        self._num_processes = num_parallel_runs

        print("[INFO]: Setting up the LLM Manager...")
        self._llm_manager = LLMManager(
            gpt_model=gpt_model,
            num_suggestions=self._num_processes,
            temperature=temperature,
            system_prompt=DIRECT_WORKFLOW_INITIAL_PROMPT,
        )

        print("[INFO]: Setting up the Task Manager...")
        self._task_manager = EurekaTaskManager(
            task=task,
            device=device,
            rl_library=rl_library,
            num_processes=self._num_processes,
            max_training_iterations=max_training_iterations,
            success_metric_string=successs_metric_string,
        )

        # Logging
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._log_dir = os.path.join(EUREKA_ROOT_DIR, "logs", "eureka", task, timestamp)
        os.makedirs(self._log_dir)
        self._tensorboard_writer = TensorboardSummaryWriter(log_dir=self._log_dir, flush_secs=10)

    def run(self, max_eureka_iterations):
        # Initial prompts
        user_prompt = DIRECT_WORKFLOW_TASK_PROMT.format(
            task_description=self._task_description,
            get_observations_method_as_string=self._task_manager.get_observations_method_as_string,
        )
        # The assistant prompt is used to feed the previous LLM output back into the LLM
        assistant_prompt = None

        # The best run across all iterations
        best_run_results = {"success_metric": None}

        for iter in range(max_eureka_iterations):
            print(f"\n{'#' * 20} Running Eureka Iteration {iter} {'#' * 20} \n")
            # Generate the GPT reward methods
            llm_outputs = self._llm_manager.prompt(user_prompt=user_prompt, assistant_prompt=assistant_prompt)
            gpt_reward_method_strings = llm_outputs["reward_strings"]
            # Log the llm outputs
            for idx, gpt_reward_method_string in enumerate(gpt_reward_method_strings):
                self._tensorboard_writer.add_text(f"Run_{idx}/raw_llm_output", llm_outputs["raw_outputs"][idx], iter)
            # Train the RL agent
            results = self._task_manager.train(gpt_reward_method_strings)
            # Evaluate the results
            iter_best_success_metric = None
            best_run_idx = 0
            for idx, result in enumerate(results):
                if not result["success"]:
                    user_feedback_prompt = TASK_FAILURE_FEEBACK_PROMPT.format(traceback_msg=result["exception"])
                else:
                    # Compute the performance metrics
                    eureka_task_feedback, success_metric_max, rewards_correlation = self._get_eureka_task_feedback(
                        result["log_dir"], self._feedback_subsampling
                    )

                    # Generate the user feedback prompt
                    user_feedback_prompt = (
                        TASK_SUCCESS_PRE_FEEDBACK_PROMT.format(feedback_subsampling=self._feedback_subsampling)
                        + eureka_task_feedback
                        + TASK_SUCCESS_POST_FEEDBACK_PROMT
                    )

                    # Store the results
                    results[idx]["eureka_task_feedback"] = eureka_task_feedback
                    results[idx]["success_metric_max"] = success_metric_max
                    results[idx]["rewards_correlation"] = rewards_correlation

                    if success_metric_max is not None and (
                        iter_best_success_metric is None or success_metric_max > iter_best_success_metric
                    ):
                        # Store the best run for this iteration
                        iter_best_success_metric = success_metric_max
                        best_run_idx = idx

                        # Store the best metric across all iterations
                        if (
                            best_run_results["success_metric"] is None
                            or iter_best_success_metric > best_run_results["success_metric"]
                        ):
                            best_run_results["success_metric"] = iter_best_success_metric
                            best_run_results["gpt_reward_method"] = gpt_reward_method_strings[idx]
                            best_run_results["task_feeback"] = eureka_task_feedback

                # Add the prompts
                results[idx]["user_prompt"] = user_feedback_prompt
                results[idx]["assistant_prompt"] = llm_outputs["raw_outputs"][idx]

            self._log_iteration_results(iter, results)

            if (
                best_run_results["success_metric"] is not None
                and best_run_results["success_metric"] >= self._successs_metric_to_win
            ):
                print(f"Task solved with success metric: {best_run_results['success_metric']}")
                break

            assistant_prompt = results[best_run_idx]["assistant_prompt"]
            user_prompt = results[best_run_idx]["user_prompt"]

        self._log_final_results(best_run_results)
        # Close the task manager
        self._task_manager.close()

    def _get_eureka_task_feedback(self, log_dir: str, feedback_subsampling: int) -> tuple[str, float, float]:
        """Get the feedback for the Eureka task.

        Args:
            log_dir: The directory where the tensorboard logs are stored.
            feedback_subsampling: The subsampling of the metrics' trajectories.
        Returns:
            A tuple containing the feedback string, the maximum of the success metric, and the correlation between the oracle and GPT rewards.
        """
        data = load_tensorboard_logs(log_dir)
        # Compute correlation between the oracle and GPT rewards
        eureka_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/eureka_total_rewards")), None)
        )
        oracle_rewards = np.array(
            next((data[key] for key in data if key.endswith("Eureka/oracle_total_rewards")), None)
        )
        rewards_correlation = np.corrcoef(eureka_rewards, oracle_rewards)[0, 1]

        success_metric_max = None
        # Make a summary of each plot in the tensorboard logs
        total_feed_back_string = ""
        for metric_name, metric_data in data.items():
            if "Eureka/" in metric_name:
                # Remove the first two data points as they are usually outliers
                metric_data = metric_data[2:]
                metric_name = metric_name.split("Eureka/", 1)[-1]
                metric_min = min(metric_data)
                metric_max = max(metric_data)
                metric_mean = sum(metric_data) / len(metric_data)
                if metric_name == "success_metric":
                    metric_name = "task_score"
                    success_metric_max = metric_max
                data_string = [f"{data:.2f}" for data in metric_data[::feedback_subsampling]]
                feedback_string = (
                    f"{metric_name}: {data_string}, Min: {metric_min:.2f}, Max: {metric_max:.2f}, Mean:"
                    f" {metric_mean:.2f} \n"
                )
                if "Eureka/success_metric" in data and metric_name == "Eureka/oracle_total_rewards":
                    # If success metric is available, we do not provide the oracle feedback
                    feedback_string = ""
                total_feed_back_string += feedback_string
        return total_feed_back_string, success_metric_max, rewards_correlation

    def _log_iteration_results(self, iter: int, results: list):
        """Log the results of the iteration."""
        for idx, result in enumerate(results):
            print(f"{'*' * 20} Run: {idx} {'*' * 20}")
            if result["success"]:
                print(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}")
            else:
                print(f"Training failed with the following exception:\n{result['exception']}\n")

        # write the iterationsresults to file
        with open(f"{self._log_dir}/eureka_iterations.txt", "a") as f:
            for idx, result in enumerate(results):
                f.write(f"{'#' * 20} Iteration: {iter} {'#' * 20}\n\n")
                f.write(f"{'*' * 20} Run: {idx} {'*' * 20}\n")
                f.write(f"- GPT reward method {result['gpt_reward_method']}\n")
                if result["success"]:
                    f.write(f"Training successful with the following metrics:\n{result['eureka_task_feedback']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", result["success_metric_max"], iter)
                else:
                    f.write(f"Training failed with the following exception:\n{result['exception']}\n")
                    self._tensorboard_writer.add_scalar(f"Run_{idx}/success_metric", 0.0, iter)
                self._tensorboard_writer.add_text(f"Run_{idx}/run_feedback", result["user_prompt"], iter)
                f.write("\n")

    def _log_final_results(self, best_run_results: dict):
        """Log the final results of the Eureka run."""
        output = ""
        if best_run_results["success_metric"] is not None:
            output += f"- Success metric: {best_run_results['success_metric']}\n"
            output += f"- GPT reward method: {best_run_results['gpt_reward_method']}\n"
            output += f"- Task metrics:\n{best_run_results['task_feeback']}\n"
        else:
            output += "- No successful training run\n"

        print("Final results:\n", output)

        with open(f"{self._log_dir}/eureka_final_result.txt", "w") as f:
            f.write(output)
