"""机器人和环境注册系统

提供无侵入式的机器人发现、注册和资源路径解析。
"""

from waleo.sim.registry.asset_resolver import AssetResolver, get_asset_resolver
from waleo.sim.registry.robot import RobotSpec, RobotRegistry, get_robot_registry

__all__ = [
    "AssetResolver",
    "get_asset_resolver",
    "RobotSpec",
    "RobotRegistry",
    "get_robot_registry",
]
