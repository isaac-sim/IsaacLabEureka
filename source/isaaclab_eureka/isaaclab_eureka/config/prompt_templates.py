# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

"""Template strings used for prompting in Isaac Lab Eureka."""


DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS = """
Your reward function should use useful variables from the environment as inputs.
It must comply to the following signature exactly:

def _get_rewards_eureka(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
    return reward, individual_rewards_dict

Make sure any new tensor or variable you introduce is on the same device as self.device.
The output of the reward function should consist of two items:
    (1) the total reward, which has a dimension of (self.num_envs,) and is a torch.Tensor,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```" and contain only the get_rewards_eureka function.

Some helpful tips for writing the reward function code:
    (1) You may find it helpful to normalize the reward to a fixed range by applying transformations like torch.exp to the overall reward or its components
    (2) If you choose to transform a reward component, then you must also introduce a temperature parameter inside the transformation function; this parameter must be a named variable in the reward function and it must not be an input variable. Each transformed reward component should have its own temperature variable
    (3) Make sure the type of each input variable is correctly specified; a float input variable should not be specified as torch.Tensor
    (4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have prefix self.). Under no circumstance can you introduce new input variables.
"""


DIRECT_WORKFLOW_INITIAL_PROMPT = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effective as possible.
Your goal is to write a reward function for the environment that will help the agent learn the task described in text.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

MANAGER_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS = """
Generate the following structure exactly:
"```python
    @configclass
    class <ClassName>:
        # existing config fields if any
        <field1>: Type = Default
        <field2>: Type = Default

        # Reward terms
        term_name1 = RewTerm(
            func=
            def func(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
                # your code here
                ...
                return reward_tensor.to(env.device)
            ,
            weight=<float>,
            params={...},
        )

        term_name2 = RewTerm(
            func=
            def func(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
                # another reward
                ...
                return reward_tensor.to(env.device)
            ,
            weight=<float>,
            params={...},
        )
...

```"

- Include as many `RewTerm` definitions as needed for the task.
- Each inline `def func` must be indented exactly 12 spaces from the file margin.
- Do not include any other methods or imports.
- Ensure each reward function returns a `(self.num_envs,)` tensor on `env.device`.
"""

MANAGER_WORKFLOW_INITIAL_PROMPT = """
You are a reward‑engineering assistant specializing in IsaacLab’s `ManagerBasedRLEnv` workflows.
Your task is to generate a complete `@configclass` definition including decorator, class header,
and one or more `RewTerm` fields.
These should be referenced in the reward functions you define.

Wrap your entire output as plain text (no markdown fences).
""" +  MANAGER_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

DIRECT_TASK_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

MANAGER_TASK_FAILURE_FEEDBACK_PROMPT = """
Executing the reward function code above has the following error: {traceback_msg}.
Please fix the bug and provide a new, improved reward function!
""" + MANAGER_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


TASK_SUCCESS_PRE_FEEDBACK_PROMPT = """
We trained a RL policy using the provided reward function code and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_subsampling} epochs and the maximum, mean, minimum values encountered:
"""

DIRECT_TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""" + DIRECT_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS

MANAGER_TASK_SUCCESS_POST_FEEDBACK_PROMPT = """
Please carefully analyze the policy feedback and provide a new, improved reward function that can better solve the task. Some helpful tips for analyzing the policy feedback:
    (1) If the success rates are always near zero, then you must rewrite the entire reward function
    (2) If the values for a certain reward component are near identical throughout, then this means RL is not able to optimize this component as it is written. You may consider
        (a) Changing its scale or the value of its temperature parameter
        (b) Re-writing the reward component
        (c) Discarding the reward component
    (3) If some reward components' magnitude is significantly larger, then you must re-scale its value to a proper range
Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
""" + MANAGER_WORKFLOW_REWARD_FORMATTING_INSTRUCTIONS


DIRECT_WORKFLOW_TASK_PROMPT = """
Write a reward function for the following task: {task_description}
The desired task score is: {success_metric_to_win}
Here is how we get the observations from the environment:
{get_observations_method_as_string}
"""

MANAGER_WORKFLOW_TASK_PROMPT = """
Write a reward function for the following task: {task_description}
The desired task score is: {success_metric_to_win}
Here is the configuration for the environment observations and the default reward, this should give you an idea of how to write the reward function:
{get_observations_method_as_string}
"""
