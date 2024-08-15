# Isaac Lab Eureka

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
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

- Install Isaac Lab, see the [installation guide](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html).

- Using a python interpreter that has Isaac Lab installed, install the Isaac Lab Eureka
    ```
    cd ext/isaaclab_eureka
    python -m pip install -e .
    ```

## Running Isaac Lab Eureka

The Openai API key has to be exposed to the script via an environment variable. We follow the Openai API convention an use ``OPENAI_API_KEY``, ``AZURE_OPENAI_API_KEY``, and ``AZURE_OPENAI_ENDPOINT``.

### Running with the Openai API

<details open>
<summary>Linux</summary>

```
OPENAI_API_KEY=your_key scripts/python train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:OPENAI_API_KEY="your_key"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```

**Command line**
```
set OPENAI_API_KEY=your_key
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```
</details>

### Running with the Azure Openai API

<details open>
<summary>Linux</summary>

```
AZURE_OPENAI_API_KEY=your_key AZURE_OPENAI_ENDPOINT=azure_endpoint_url python scripts/train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```
</details>

<details>
<summary>Windows</summary>

**Powershell**
```
$env:AZURE_OPENAI_API_KEY="your_key"
$env:AZURE_OPENAI_ENDPOINT="azure_endpoint_url"
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```

**Command line**
```
set AZURE_OPENAI_API_KEY=your_key
set AZURE_OPENAI_ENDPOINT=azure_endpoint_url
python scripts\train.py --task=Isaac-Cartpole-Direct-v0 --max_training_iterations=40 --rl_library="rl_games"
```
</details>

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
