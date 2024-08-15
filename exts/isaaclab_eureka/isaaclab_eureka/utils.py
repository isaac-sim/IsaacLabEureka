# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: MIT

import os
import sys
from collections import defaultdict

import GPUtil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(path: str):
    """Load tensorboard logs from a given path.

    Args:
        path: The path to the tensorboard logs.

    Returns:
        A dictionary with the tags and their respective values.
    """
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)

    return data


def get_freest_gpu():
    """Get the GPU with the most free memory."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    # Sort GPUs by memory usage
    gpus.sort(key=lambda gpu: gpu.memoryUsed)
    return gpus[0].id


class MuteOutput:
    """Context manager to mute stdout and stderr."""

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
