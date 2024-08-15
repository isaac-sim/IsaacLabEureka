# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: MIT

TASKS_CFG = {
    "Isaac-Cartpole-Direct-v0": {
        "description": "balance a pole on a cart so that the pole stays upright",
        "successs_metric": "self.episode_length_buf[env_ids].float().mean() / self.max_episode_length",
        "successs_metric_to_win": 0.99,
    },
}
"""Configuration for the tasks supported by Isaac Lab Eureka.

`TASKS_CFG` is a dictionary that maps task names to their configuration. Each task configuration
is a dictionary that contains the following keys:

- `description`: A description of the task.
- `success_metric`: A Python expression that computes the success metric for the task.
- `success_metric_to_win`: The threshold for the success metric to win the task and stop.

"""
