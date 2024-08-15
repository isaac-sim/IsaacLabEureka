# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import unittest

from isaaclab_eureka.config import TASKS_CFG
from isaaclab_eureka.managers import EurekaTaskManager
from isaaclab_eureka.utils import load_tensorboard_logs

REWARD_FUNCTION_CORRECT = """
def _get_rewards_eureka(self):
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
    alive_rew = 1.0 - self.reset_terminated.float()
    pole_rew = -torch.square(pole_pos)
    cart_rew = -0.05*torch.abs(cart_pos)
    reward = alive_rew + pole_rew + cart_rew
    return reward, {"alive_rew": alive_rew, "pole_rew": pole_rew, "cart_rew": cart_rew}
"""


REWARD_FUNCTION_BAD_CONVERGENCE = """
def _get_rewards_eureka(self):
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
    pole_rew = torch.square(pole_pos)
    cart_rew = torch.abs(cart_pos)
    reward = pole_rew + cart_rew
    return reward, {"pole_rew": pole_rew, "cart_rew": cart_rew}
"""


REWARD_FUNCTION_RAISE_RUNTIME_ERROR = """
def _get_rewards_eureka(self):
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
    pole_rew = -0*torch.square(pole_pos)
    cart_rew = -0.0*torch.abs(cart_pos)
    reward = alive_rew + pole_rew + cart_rew
    return reward, {"alive_rew": alive_rew, "pole_rew": pole_rew, "cart_rew": cart_rew}
"""


REWARD_FUNCTION_RAISE_WRONG_SIGNATURE_ERROR = """
def get_rewards_eureka(self):
    pole_pos = self.joint_pos[:, self._pole_dof_idx[0]]
    cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
    alive_rew = 1.0 - self.reset_terminated.float()
    pole_rew = -torch.square(pole_pos)
    cart_rew = -0.05*torch.abs(cart_pos)
    reward = alive_rew + pole_rew + cart_rew
    return reward, {"alive_rew": alive_rew, "pole_rew": pole_rew, "cart_rew": cart_rew}
"""


class TestTaskManager(unittest.TestCase):
    """Test fixture for the Eureka Task manager."""

    @classmethod
    def setUpClass(cls):
        """Set up the test fixture."""
        task = "Isaac-Cartpole-Direct-v0"
        cls.num_parallel_runs = 1
        cls._task_manager = EurekaTaskManager(
            task=task,
            device="cuda",
            rl_library="rl_games",
            num_processes=cls.num_parallel_runs,
            max_training_iterations=100,
            success_metric_string=TASKS_CFG[task].get("successs_metric", ""),
        )

    @classmethod
    def tearDownClass(cls):
        """Tear down the test fixture."""
        cls._task_manager.close()

    def test_pass_and_fail(self):
        """Test that one run succeeds while the other fails."""
        results = self._task_manager.train([REWARD_FUNCTION_CORRECT, REWARD_FUNCTION_RAISE_RUNTIME_ERROR])
        # Check the success flag
        self.assertTrue(results[0]["success"])
        data = load_tensorboard_logs(results[0]["log_dir"])
        # Check the success metric
        success_metric = next((data[key] for key in data if key.endswith("Eureka/success_metric")), None)
        self.assertGreater(max(success_metric[2:]), 0.99)
        # Check the success flag
        self.assertFalse(results[1]["success"])
        # Check if the exception was reported
        self.assertTrue("exception" in results[1])

    def test_bad_convergence(self):
        """Test bad convergence."""
        results = self._task_manager.train([REWARD_FUNCTION_BAD_CONVERGENCE] * self.num_parallel_runs)
        for result in results:
            # Check the success flag
            self.assertTrue(result["success"])
            data = load_tensorboard_logs(result["log_dir"])
            # Check the success metric
            success_metric = next((data[key] for key in data if key.endswith("Eureka/success_metric")), None)
            self.assertLess(max(success_metric[2:]), 0.4)

    def test_raise_wrong_signature_error(self):
        """Test raise wrong signature error."""
        results = self._task_manager.train([REWARD_FUNCTION_RAISE_WRONG_SIGNATURE_ERROR] * self.num_parallel_runs)
        # Check the success flag
        for result in results:
            # Check the success flag
            self.assertFalse(result["success"])
            # Check if the exception was reported
            self.assertTrue(
                result["exception"]
                == "The reward function must be a string that starts with 'def _get_rewards_eureka(self)'."
            )

    def test_runs_correctly_after_fail(self):
        """Test if the task runs correctly after a failed run."""
        results = self._task_manager.train([REWARD_FUNCTION_RAISE_RUNTIME_ERROR] * self.num_parallel_runs)
        for result in results:
            # Check the success flag
            self.assertFalse(result["success"])
            # Check if the exception was reported
            self.assertTrue("exception" in result)
        # Run the task again
        results = self._task_manager.train([REWARD_FUNCTION_CORRECT] * self.num_parallel_runs)
        for result in results:
            # Check the success flag
            self.assertTrue(result["success"])
            data = load_tensorboard_logs(result["log_dir"])
            # Check the success metric
            success_metric = next((data[key] for key in data if key.endswith("Eureka/success_metric")), None)
            self.assertGreater(max(success_metric[2:]), 0.99)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=True)
