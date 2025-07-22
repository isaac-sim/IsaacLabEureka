# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import is_dataclass
import os
import sys
from collections import defaultdict

import inspect
import GPUtil

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def turn_cfg_to_string(cfg, use=None, tab_level=0):
    cfg_string = ""
    tab_string = "    " * tab_level
    if isinstance(cfg, dict):
        cfg_string += "{\n"
        for key, value in cfg.items():
            if use is not None and key not in use:
                continue
            if value is None or value == {} or value == []:
                continue
            cfg_string += f"{tab_string}    {key}:{turn_cfg_to_string(value, None, tab_level + 1)},\n"
        cfg_string += f"{tab_string}}}"
    elif is_dataclass(cfg):
        cfg_string += f"{cfg.__class__.__name__}(\n"
        for field in cfg.__dataclass_fields__.values():
            value = getattr(cfg, field.name)
            if use != None and field.name not in use:
                continue
            if value is None or value == {} or value == []:
                continue
            cfg_string += f"{tab_string}    {field.name}={turn_cfg_to_string(value, None, tab_level + 1)},\n"
        cfg_string += f"{tab_string})"
    elif inspect.isfunction(cfg):
        cfg_string += f"```\n{inspect.getsource(cfg).strip()}\n{tab_string}```"
    elif isinstance(cfg, str):
        cfg_string += f"\"{cfg}\""
    else:
        cfg_string += f"{cfg}"

    return cfg_string

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
