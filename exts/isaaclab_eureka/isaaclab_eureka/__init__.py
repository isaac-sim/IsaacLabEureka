# Copyright (c) 2024, The Isaac Lab Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import os

EUREKA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 3)

from .eureka import Eureka
