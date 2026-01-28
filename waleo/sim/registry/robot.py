"""机器人注册系统

提供插件式的机器人发现和注册机制，支持从配置文件加载机器人规格。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Any
import warnings

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    warnings.warn(
        "PyYAML not installed. Robot YAML configs will not be loaded. "
        "Install with: pip install pyyaml",
        ImportWarning
    )

from waleo.sim.registry.asset_resolver import get_asset_resolver


@dataclass
class RobotSpec:
    """机器人规格定义

    包含机器人的基本信息、路径和任务特定配置。

    Attributes:
        name: 机器人名称
        urdf_path: URDF 文件路径
        urdf_config: URDF 加载配置（材质、碰撞参数等）
        robot_dir: 机器人资源目录
        config_path: 配置文件路径（如果从 YAML 加载）
        dof: 自由度数量
        default_kwargs: 后端特定默认参数（通用）
        task_configs: 任务特定配置字典
        metadata: 其他元数据

    Examples:
        >>> spec = RobotSpec(
        ...     name="rj2506",
        ...     urdf_path=Path("/path/to/rj2506.urdf"),
        ...     dof=8,
        ...     default_kwargs={"control_mode": "pd_joint_pos"},  # ManiSkill
        ... )
    """

    name: str
    urdf_path: Path
    robot_dir: Path

    # 可选属性
    urdf_config: Dict[str, Any] = field(default_factory=dict)
    config_path: Optional[Path] = None
    dof: int = 0
    default_kwargs: Dict[str, Any] = field(default_factory=dict)
    task_configs: Dict[str, Dict] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "RobotSpec":
        """从 YAML 配置文件加载机器人规格

        Args:
            config_path: YAML 配置文件路径

        Returns:
            RobotSpec 实例

        Raises:
            ImportError: 如果 PyYAML 未安装
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置格式无效

        YAML 格式示例:
            name: rj2506
            urdf: urdf/rj2506.urdf  # 相对于机器人目录
            dof: 8

            # 后端特定默认参数（通用）
            default_kwargs:
              control_mode: pd_joint_pos  # ManiSkill 特定
              frame_skip: 5               # MuJoCo 特定（示例）

            urdf_config:
              materials:
                gripper:
                  static_friction: 2.0
                  dynamic_friction: 2.0

            task_configs:
              PickCube-v1:
                robot_pose:
                  offset: [-0.85, 0, -0.35]
                object_config:
                  cube_half_size: 0.008
        """
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required to load robot configs from YAML")

        if not config_path.exists():
            raise FileNotFoundError(f"Robot config not found: {config_path}")

        with open(config_path) as f:
            data = yaml.safe_load(f)

        if not data or "name" not in data:
            raise ValueError(f"Invalid robot config: {config_path}")

        robot_name = data["name"]
        resolver = get_asset_resolver()

        # 解析 URDF 路径
        urdf_filename = data.get("urdf", f"{robot_name}.urdf")
        if "/" in urdf_filename or "\\" in urdf_filename:
            # 相对路径，相对于配置文件所在的机器人目录
            urdf_path = config_path.parent / urdf_filename
        else:
            # 只是文件名，通过 resolver 查找
            urdf_path = resolver.resolve_urdf(robot_name, urdf_filename)

        if urdf_path is None or not urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found for robot '{robot_name}': {urdf_filename}"
            )

        # 解析机器人目录
        robot_dir = resolver.resolve_robot_dir(robot_name)
        if robot_dir is None:
            robot_dir = config_path.parent

        # 处理 default_kwargs（支持两种格式）
        # 新格式：default_kwargs: {control_mode: pd_joint_pos}
        # 旧格式：control_mode: pd_joint_pos（向后兼容）
        default_kwargs = data.get("default_kwargs", {})

        # 向后兼容：如果 YAML 中直接写了 control_mode，自动迁移到 default_kwargs
        if "control_mode" in data and "control_mode" not in default_kwargs:
            default_kwargs["control_mode"] = data["control_mode"]

        return cls(
            name=robot_name,
            urdf_path=urdf_path,
            robot_dir=robot_dir,
            urdf_config=data.get("urdf_config", {}),
            config_path=config_path,
            dof=data.get("dof", 0),
            default_kwargs=default_kwargs,
            task_configs=data.get("task_configs", {}),
            metadata=data.get("metadata", {}),
        )

    def get_task_config(self, task_id: str) -> Optional[Dict]:
        """获取特定任务的配置

        Args:
            task_id: 任务 ID（如 "PickCube-v1"）

        Returns:
            任务配置字典，如果未配置则返回 None
        """
        return self.task_configs.get(task_id)


class RobotRegistry:
    """机器人注册中心

    自动发现和管理可用的机器人规格。支持：
    - 从 WALEO_ASSETS_DIR 自动发现机器人
    - 从 YAML 配置加载机器人规格
    - 手动注册机器人

    Examples:
        >>> registry = RobotRegistry()
        >>> robots = registry.list_robots()
        >>> spec = registry.get("rj2506")
        >>> print(spec.urdf_path)
    """

    def __init__(self, auto_discover: bool = True):
        """初始化机器人注册中心

        Args:
            auto_discover: 是否自动发现机器人（默认 True）
        """
        self.resolver = get_asset_resolver()
        self._robots: Dict[str, RobotSpec] = {}

        if auto_discover:
            self._discover_robots()

    def _discover_robots(self):
        """自动发现机器人

        扫描所有搜索路径，查找包含 robot.yaml 的机器人目录。
        """
        for search_path in self.resolver.search_paths:
            robots_dir = search_path / "robots"
            if not robots_dir.exists():
                continue

            for robot_dir in robots_dir.iterdir():
                if not robot_dir.is_dir():
                    continue

                # 查找 robot.yaml 配置文件
                config_file = robot_dir / "robot.yaml"
                if config_file.exists() and YAML_AVAILABLE:
                    try:
                        spec = RobotSpec.from_yaml(config_file)
                        self.register(spec)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to load robot config from {config_file}: {e}",
                            UserWarning
                        )
                else:
                    # 没有 YAML 配置，尝试创建基本的 RobotSpec
                    urdf_path = self.resolver.resolve_urdf(robot_dir.name)
                    if urdf_path:
                        spec = RobotSpec(
                            name=robot_dir.name,
                            urdf_path=urdf_path,
                            robot_dir=robot_dir,
                        )
                        self.register(spec)

    def register(self, spec: RobotSpec):
        """注册机器人规格

        Args:
            spec: RobotSpec 实例

        Examples:
            >>> spec = RobotSpec(name="my_robot", urdf_path=Path(...), robot_dir=Path(...))
            >>> registry.register(spec)
        """
        self._robots[spec.name] = spec

    def get(self, name: str) -> Optional[RobotSpec]:
        """获取机器人规格

        Args:
            name: 机器人名称

        Returns:
            RobotSpec 实例，如果未找到则返回 None

        Examples:
            >>> spec = registry.get("rj2506")
            >>> if spec:
            ...     print(f"Found robot at {spec.urdf_path}")
        """
        return self._robots.get(name)

    def list_robots(self) -> List[str]:
        """列出所有已注册的机器人

        Returns:
            机器人名称列表

        Examples:
            >>> robots = registry.list_robots()
            >>> print(robots)
            ['rj2506', 'panda', 'fetch']
        """
        return sorted(list(self._robots.keys()))

    def is_registered(self, name: str) -> bool:
        """检查机器人是否已注册

        Args:
            name: 机器人名称

        Returns:
            如果已注册返回 True，否则返回 False
        """
        return name in self._robots

    def unregister(self, name: str) -> bool:
        """注销机器人

        Args:
            name: 机器人名称

        Returns:
            如果成功注销返回 True，如果机器人不存在返回 False
        """
        if name in self._robots:
            del self._robots[name]
            return True
        return False


# 全局单例实例
_robot_registry = None


def get_robot_registry() -> RobotRegistry:
    """获取全局机器人注册中心（单例）

    Returns:
        RobotRegistry 实例

    Examples:
        >>> from waleo.sim.registry import get_robot_registry
        >>> registry = get_robot_registry()
        >>> spec = registry.get("rj2506")
    """
    global _robot_registry
    if _robot_registry is None:
        _robot_registry = RobotRegistry()
    return _robot_registry
