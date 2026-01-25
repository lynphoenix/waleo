"""环境包装器

提供无侵入式的环境定制包装器。
"""

from waleo.sim.wrappers.custom_robot import (
    CustomRobotWrapper,
    TaskConfigWrapper,
    CameraConfigWrapper,
    create_wrapped_env,
)

__all__ = [
    "CustomRobotWrapper",
    "TaskConfigWrapper",
    "CameraConfigWrapper",
    "create_wrapped_env",
]
