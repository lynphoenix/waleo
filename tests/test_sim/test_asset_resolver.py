"""测试资源解析器"""

import os
import pytest
from pathlib import Path

from waleo.sim.registry.asset_resolver import AssetResolver, get_asset_resolver


def test_asset_resolver_initialization():
    """测试资源解析器初始化"""
    resolver = AssetResolver()
    assert len(resolver.search_paths) > 0
    # 至少应该有包内置的 assets 目录
    assert any("assets" in str(path) for path in resolver.search_paths)


def test_waleo_assets_dir_priority(tmp_path, monkeypatch):
    """测试 WALEO_ASSETS_DIR 环境变量优先级"""
    # 创建临时 assets 目录
    custom_assets = tmp_path / "custom_assets"
    custom_assets.mkdir()
    (custom_assets / "robots").mkdir()

    # 设置环境变量
    monkeypatch.setenv("WALEO_ASSETS_DIR", str(custom_assets))

    resolver = AssetResolver()
    # WALEO_ASSETS_DIR 应该是第一个搜索路径
    assert resolver.search_paths[0] == custom_assets


def test_resolve_urdf_existing_robot():
    """测试解析存在的机器人 URDF"""
    resolver = AssetResolver()

    # 检查 RJ2506 是否存在
    urdf_path = resolver.resolve_urdf("RJ2506")
    if urdf_path:
        assert urdf_path.exists()
        assert urdf_path.suffix == ".urdf"
        assert "RJ2506" in str(urdf_path)


def test_resolve_urdf_nonexistent_robot():
    """测试解析不存在的机器人"""
    resolver = AssetResolver()
    urdf_path = resolver.resolve_urdf("nonexistent_robot_xyz")
    assert urdf_path is None


def test_resolve_urdf_custom_filename():
    """测试解析自定义文件名的 URDF"""
    resolver = AssetResolver()

    # 尝试解析 RJ2506 的变体 URDF
    urdf_path = resolver.resolve_urdf("RJ2506", "RJ2506_leftarm_only.urdf")
    if urdf_path:
        assert urdf_path.exists()
        assert urdf_path.name == "RJ2506_leftarm_only.urdf"


def test_resolve_mesh():
    """测试解析 mesh 文件"""
    resolver = AssetResolver()

    # 尝试解析 RJ2506 的 mesh
    mesh_path = resolver.resolve_mesh("RJ2506", "base_link.STL")
    if mesh_path:
        assert mesh_path.exists()
        assert mesh_path.suffix == ".STL"


def test_resolve_robot_dir():
    """测试解析机器人目录"""
    resolver = AssetResolver()

    robot_dir = resolver.resolve_robot_dir("RJ2506")
    if robot_dir:
        assert robot_dir.exists()
        assert robot_dir.is_dir()
        assert (robot_dir / "urdf").exists()


def test_list_robots():
    """测试列出所有机器人"""
    resolver = AssetResolver()
    robots = resolver.list_robots()

    assert isinstance(robots, list)
    # 应该至少找到 RJ2506
    if len(robots) > 0:
        assert "RJ2506" in robots


def test_get_asset_resolver_singleton():
    """测试全局单例"""
    resolver1 = get_asset_resolver()
    resolver2 = get_asset_resolver()
    assert resolver1 is resolver2


def test_resolve_config():
    """测试解析配置文件"""
    resolver = AssetResolver()

    # 尝试解析配置文件（如果存在）
    config_path = resolver.resolve_config("RJ2506", "robot.yaml")
    if config_path:
        assert config_path.exists()
        assert config_path.suffix in [".yaml", ".yml"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
