"""资源路径解析器

支持通过 WALEO_ASSETS_DIR 环境变量发现机器人和环境资源，
实现无侵入式的自定义机器人集成。
"""

import os
from pathlib import Path
from typing import Optional, List
import warnings


class AssetResolver:
    """资源路径解析器

    通过 WALEO_ASSETS_DIR 环境变量和默认路径解析机器人资源文件（URDF、mesh 等）。

    搜索顺序（从高到低优先级）：
    1. WALEO_ASSETS_DIR 环境变量指定的目录
    2. 包内置 assets 目录
    3. 系统级 assets 目录

    Examples:
        >>> resolver = AssetResolver()
        >>> urdf_path = resolver.resolve_urdf("rj2506")
        >>> print(urdf_path)
        /path/to/waleo/assets/robots/rj2506/urdf/rj2506.urdf

        >>> # 设置环境变量
        >>> import os
        >>> os.environ['WALEO_ASSETS_DIR'] = '/custom/assets'
        >>> resolver = AssetResolver()  # 会优先搜索 /custom/assets
    """

    def __init__(self):
        """初始化资源解析器"""
        self.search_paths = self._get_search_paths()
        self._log_search_paths()

    def _get_search_paths(self) -> List[Path]:
        """获取有序的资源搜索路径列表

        Returns:
            按优先级排序的路径列表
        """
        paths = []

        # 1. 用户指定的 WALEO_ASSETS_DIR（最高优先级）
        if "WALEO_ASSETS_DIR" in os.environ:
            custom_dir = Path(os.environ["WALEO_ASSETS_DIR"])
            if custom_dir.exists():
                paths.append(custom_dir)
            else:
                warnings.warn(
                    f"WALEO_ASSETS_DIR points to non-existent directory: {custom_dir}",
                    UserWarning
                )

        # 2. 包相对 assets 目录
        package_assets = Path(__file__).parent.parent.parent.parent / "assets"
        if package_assets.exists():
            paths.append(package_assets)

        # 3. 系统级 assets（如果存在）
        system_assets = Path("/usr/local/share/waleo/assets")
        if system_assets.exists():
            paths.append(system_assets)

        return paths

    def _log_search_paths(self):
        """记录搜索路径（用于调试）"""
        if os.environ.get("WALEO_DEBUG"):
            print(f"AssetResolver initialized with {len(self.search_paths)} search paths:")
            for i, path in enumerate(self.search_paths, 1):
                print(f"  {i}. {path}")

    def resolve_urdf(self, robot_name: str, urdf_filename: Optional[str] = None) -> Optional[Path]:
        """解析机器人 URDF 文件路径

        Args:
            robot_name: 机器人名称（如 "rj2506"）
            urdf_filename: URDF 文件名，默认为 {robot_name}.urdf

        Returns:
            URDF 文件的绝对路径，如果未找到则返回 None

        Examples:
            >>> resolver = AssetResolver()
            >>> path = resolver.resolve_urdf("rj2506")
            >>> path = resolver.resolve_urdf("rj2506", "rj2506_leftarm_only.urdf")
        """
        if urdf_filename is None:
            urdf_filename = f"{robot_name}.urdf"

        for search_path in self.search_paths:
            urdf_path = search_path / "robots" / robot_name / "urdf" / urdf_filename
            if urdf_path.exists():
                return urdf_path

        return None

    def resolve_mesh(self, robot_name: str, mesh_filename: str) -> Optional[Path]:
        """解析机器人 mesh 文件路径

        Args:
            robot_name: 机器人名称
            mesh_filename: Mesh 文件名（如 "arm_link1.stl"）

        Returns:
            Mesh 文件的绝对路径，如果未找到则返回 None

        Examples:
            >>> resolver = AssetResolver()
            >>> path = resolver.resolve_mesh("rj2506", "left_arm_link0.STL")
        """
        for search_path in self.search_paths:
            mesh_path = search_path / "robots" / robot_name / "meshes" / mesh_filename
            if mesh_path.exists():
                return mesh_path

        return None

    def resolve_robot_dir(self, robot_name: str) -> Optional[Path]:
        """解析机器人目录路径

        Args:
            robot_name: 机器人名称

        Returns:
            机器人目录的绝对路径，如果未找到则返回 None
        """
        for search_path in self.search_paths:
            robot_dir = search_path / "robots" / robot_name
            if robot_dir.exists():
                return robot_dir

        return None

    def resolve_config(self, robot_name: str, config_filename: str = "robot.yaml") -> Optional[Path]:
        """解析机器人配置文件路径

        Args:
            robot_name: 机器人名称
            config_filename: 配置文件名，默认为 "robot.yaml"

        Returns:
            配置文件的绝对路径，如果未找到则返回 None
        """
        for search_path in self.search_paths:
            config_path = search_path / "robots" / robot_name / config_filename
            if config_path.exists():
                return config_path

        return None

    def list_robots(self) -> List[str]:
        """列出所有可发现的机器人

        Returns:
            机器人名称列表

        Examples:
            >>> resolver = AssetResolver()
            >>> robots = resolver.list_robots()
            >>> print(robots)
            ['rj2506', 'panda', 'fetch']
        """
        robots = set()

        for search_path in self.search_paths:
            robots_dir = search_path / "robots"
            if not robots_dir.exists():
                continue

            for robot_dir in robots_dir.iterdir():
                if robot_dir.is_dir():
                    # 检查是否有 URDF 文件
                    urdf_dir = robot_dir / "urdf"
                    if urdf_dir.exists() and any(urdf_dir.glob("*.urdf")):
                        robots.add(robot_dir.name)

        return sorted(list(robots))


# 全局单例实例
_asset_resolver = None


def get_asset_resolver() -> AssetResolver:
    """获取全局资源解析器实例（单例）

    Returns:
        AssetResolver 实例

    Examples:
        >>> from waleo.sim.registry import get_asset_resolver
        >>> resolver = get_asset_resolver()
        >>> urdf_path = resolver.resolve_urdf("rj2506")
    """
    global _asset_resolver
    if _asset_resolver is None:
        _asset_resolver = AssetResolver()
    return _asset_resolver
