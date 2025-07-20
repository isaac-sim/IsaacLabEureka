# Isaac Lab Eureka

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository is an implementation of *[Eureka](https://github.com/eureka-research/Eureka): Human-Level Reward Design via Coding Large Language Models* in Isaac Lab.
It prompts an LLM to discover and tune reward functions automatically for your specific task.

We support the native Openai and the Azure Openai APIs.

## Installation

- Make sure that you have either an [Openai API](https://platform.openai.com/api-keys) or [Azure Openai API](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line%2Cpython-new&pivots=programming-language-python) key.

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

- Using a python interpreter that has Isaac Lab installed, install Isaac Lab Eureka
    ```
    python -m pip install -e source/isaaclab_eureka
    ```

## Running Isaac Lab Eureka

Run Eureka from the root repo directory ``IsaacLabEureka``.

The Openai API key has to be exposed to the script via an environment variable. We follow the Openai API convention and use ``OPENAI_API_KEY``, ``AZURE_OPENAI_API_KEY``, and ``AZURE_OPENAI_ENDPOINT``.

### Running with the Openai API

<details open>
<summary>Linux</summary>

```
OPENAI_API_KEY=your_key python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:OPENAI_API_KEY="your_key"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```

**Command line**
```
set OPENAI_API_KEY=your_key
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

### Running with the Azure Openai API

<details open>
<summary>Linux</summary>

```
AZURE_OPENAI_API_KEY=your_key AZURE_OPENAI_ENDPOINT=azure_endpoint_url python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:AZURE_OPENAI_API_KEY="your_key"
$env:AZURE_OPENAI_ENDPOINT="azure_endpoint_url"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```

**Command line**
```
set AZURE_OPENAI_API_KEY=your_key
set AZURE_OPENAI_ENDPOINT=azure_endpoint_url
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=100 --rl_library="rl_games"
```
</details>

### Running Eureka Trained Policies

For each Eureka run, logs for the Eureka iterations are available under ``IsaacLabEureka/logs/eureka``.
This directory holds files containing the output from each Eureka iteration, as well as output and metrics
of the final Eureka results for the task. The tensorboard log also contains a Text tab which shows the raw LLM output
and the provided feedback at every iteration.

In addition, trained policies during the Eureka run are saved under ``IsaacLabEureka/logs/rl_runs``.
This directory contains checkpoints for each valid Eureka run, similar to the checkpoints available
when training with Isaac Lab.

To run inference on an Eureka-trained policy, locate the path to the desired checkpoint and run the ``scripts/play.py`` script.

For RSL RL, run:

```
    python scripts/play.py --task=Isaac-Cartpole-Direct-v0 --checkpoint=/path/to/desired/checkpoint.pt --num_envs=20 --rl_library="rsl_rl"
```

For RL-Games, run:

```
    python scripts/play.py --task=Isaac-Cartpole-Direct-v0 --checkpoint=/path/to/desired/checkpoint.pth --num_envs=20 --rl_library="rl_games"
```

### Limitations

- Isaac Lab Eureka currently only supports tasks implemented in the direct-workflow style, basing off of the ``DirectRLEnv`` class.
Available examples can be found in the [task config](source/isaaclab_eureka/isaaclab_eureka/config/tasks.py). Following the ``DirectRLEnv``
interface, we assume each task has the observation function implemented in a method named ``_get_observations()``.
- Currently, only RSL RL and RL-Games libraries are supported.
- Due to limitations of multiprocessing on Windows, running with argument ``num_parallel_runs`` > 1 is not supported on Windows.
- When running with ``num_parallel_runs > 1`` on a single-GPU machine, training will run in parallel in the background and CPU and memory usage will increase.
- Best policy is selected based on the ``success_metric`` defined for the task. For best performance, make sure to define an accurate success metric in the task config to guide the reward function generation process.
- During the reward generation process, the LLM may generate code that introduces syntax or logical errors during the training process. In such case, the error message will be propagated to the output and the Eureka iteration will be skipped.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
