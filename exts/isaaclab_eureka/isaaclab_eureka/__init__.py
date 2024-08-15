# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: MIT

import os

EUREKA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 3)

from .eureka import Eureka
