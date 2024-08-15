# Copyright (c) 2022-2024, The IsaacLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

EUREKA_ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), *[".."] * 3)

from .eureka import Eureka
