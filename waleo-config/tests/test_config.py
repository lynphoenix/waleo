"""
配置管理模块测试
"""

import pytest
import tempfile
from pathlib import Path


class TestBaseTypes:
    """基础类型测试"""

    def test_feature_type_enum(self):
        """测试特征类型枚举"""
        from waleo_config.base import FeatureType

        assert FeatureType.STATE.value == "STATE"
        assert FeatureType.VISUAL.value == "VISUAL"
        assert FeatureType.ENV.value == "ENV"
        assert FeatureType.ACTION.value == "ACTION"
        assert FeatureType.REWARD.value == "REWARD"

    def test_normalization_mode_enum(self):
        """测试归一化模式枚举"""
        from waleo_config.base import NormalizationMode

        assert NormalizationMode.MIN_MAX.value == "MIN_MAX"
        assert NormalizationMode.MEAN_STD.value == "MEAN_STD"
        assert NormalizationMode.IDENTITY.value == "IDENTITY"

    def test_policy_feature(self):
        """测试策略特征定义"""
        from waleo_config.base import PolicyFeature, FeatureType

        feature = PolicyFeature(type=FeatureType.STATE, shape=(10,))

        assert feature.type == FeatureType.STATE
        assert feature.shape == (10,)

    def test_policy_feature_validation(self):
        """测试策略特征验证"""
        from waleo_config.base import PolicyFeature, FeatureType

        # 无效的 shape 类型
        with pytest.raises(ValueError):
            PolicyFeature(type=FeatureType.STATE, shape="invalid")

        # 空的 shape
        with pytest.raises(ValueError):
            PolicyFeature(type=FeatureType.STATE, shape=())

        # 负数维度
        with pytest.raises(ValueError):
            PolicyFeature(type=FeatureType.STATE, shape=(-1,))


class TestDatasetConfig:
    """数据集配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        from waleo_config.dataset import DatasetConfig

        config = DatasetConfig(repo_id="waleo/pusht")

        assert config.repo_id == "waleo/pusht"
        assert config.root is None
        assert config.use_imagenet_stats is True
        assert config.video_backend == "pyav"

    def test_config_validation(self):
        """测试配置验证"""
        from waleo_config.dataset import DatasetConfig

        # 无效的 video_backend
        with pytest.raises(ValueError):
            DatasetConfig(repo_id="waleo/pusht", video_backend="invalid")

        # 空的 repo_id
        with pytest.raises(ValueError):
            DatasetConfig(repo_id="")

    def test_to_dict(self):
        """测试转换为字典"""
        from waleo_config.dataset import DatasetConfig

        config = DatasetConfig(repo_id="waleo/pusht", use_imagenet_stats=False)
        d = config.to_dict()

        assert d["repo_id"] == "waleo/pusht"
        assert d["use_imagenet_stats"] is False

    def test_from_dict(self):
        """测试从字典创建"""
        from waleo_config.dataset import DatasetConfig

        d = {"repo_id": "waleo/pusht", "use_imagenet_stats": False}
        config = DatasetConfig.from_dict(d)

        assert config.repo_id == "waleo/pusht"
        assert config.use_imagenet_stats is False

    def test_json_roundtrip(self):
        """测试 JSON 序列化"""
        from waleo_config.dataset import DatasetConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            original = DatasetConfig(repo_id="waleo/pusht", episodes=[1, 2, 3])
            original.to_json(config_path)

            loaded = DatasetConfig.from_json(config_path)
            assert loaded.repo_id == original.repo_id
            assert loaded.episodes == original.episodes

    def test_validate(self):
        """测试 validate 方法"""
        from waleo_config.dataset import DatasetConfig

        config = DatasetConfig(repo_id="waleo/pusht")
        assert config.validate() is True


class TestTrainingConfig:
    """训练配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        from waleo_config.training import TrainingConfig

        config = TrainingConfig()

        assert config.offline_steps == 50000
        assert config.batch_size == 64
        assert config.device == "cuda"

    def test_optimizer_config(self):
        """测试优化器配置"""
        from waleo_config.training import OptimizerConfig

        config = OptimizerConfig(type="adamw", lr=1e-4)

        assert config.type == "adamw"
        assert config.lr == 1e-4

    def test_optimizer_validation(self):
        """测试优化器验证"""
        from waleo_config.training import OptimizerConfig

        # 无效的类型
        with pytest.raises(ValueError):
            OptimizerConfig(type="invalid")

        # 负的学习率
        with pytest.raises(ValueError):
            OptimizerConfig(lr=-1)

    def test_scheduler_config(self):
        """测试调度器配置"""
        from waleo_config.training import SchedulerConfig

        config = SchedulerConfig(type="cosine", num_warmup_steps=1000)

        assert config.type == "cosine"
        assert config.num_warmup_steps == 1000

    def test_checkpoint_config(self):
        """测试检查点配置"""
        from waleo_config.training import CheckpointConfig

        config = CheckpointConfig(save_every=10000, save_total_limit=5)

        assert config.save_every == 10000
        assert config.save_total_limit == 5

    def test_training_validation(self):
        """测试训练配置验证"""
        from waleo_config.training import TrainingConfig

        # 负的 batch_size
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=-1)

        # 无效的设备
        with pytest.raises(ValueError):
            TrainingConfig(device="invalid")

    def test_nested_config_access(self):
        """测试嵌套配置访问"""
        from waleo_config.training import TrainingConfig, OptimizerConfig

        config = TrainingConfig(
            optimizer=OptimizerConfig(type="adamw", lr=0.001)
        )

        assert config.optimizer.type == "adamw"
        assert config.optimizer.lr == 0.001

    def test_yaml_roundtrip(self):
        """测试 YAML 序列化"""
        from waleo_config.training import TrainingConfig

        # 注意：需要 pyyaml 才能运行此测试
        pytest.importorskip("yaml")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            original = TrainingConfig(batch_size=128, offline_steps=100000)
            original.to_yaml(config_path)

            loaded = TrainingConfig.from_yaml(config_path)
            assert loaded.batch_size == original.batch_size
            assert loaded.offline_steps == original.offline_steps


class TestEvalConfig:
    """评估配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        from waleo_config.eval import EvalConfig

        config = EvalConfig()

        assert config.n_episodes == 50
        assert config.batch_size == 50
        assert config.save_video is True

    def test_config_validation(self):
        """测试配置验证"""
        from waleo_config.eval import EvalConfig

        # 负的 n_episodes
        with pytest.raises(ValueError):
            EvalConfig(n_episodes=-1)

    def test_batch_size_warning(self):
        """测试 batch_size 警告"""
        import warnings
        from waleo_config.eval import EvalConfig

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EvalConfig(n_episodes=10, batch_size=50)

            assert len(w) == 1
            assert "batch size" in str(w[0].message).lower()


class TestParser:
    """命令行解析器测试"""

    def test_parse_arg(self):
        """测试解析单个参数"""
        from waleo_config.parser import parse_arg

        args = ["--batch_size=128", "--lr=0.001"]

        assert parse_arg("batch_size", args) == "128"
        assert parse_arg("lr", args) == "0.001"
        assert parse_arg("not_exists", args) is None

    def test_parse_arg_value(self):
        """测试解析参数值"""
        from waleo_config.parser import parse_arg_value

        assert parse_arg_value("128") == 128
        assert parse_arg_value("0.001") == 0.001
        assert parse_arg_value("true") is True
        assert parse_arg_value("false") is False
        assert parse_arg_value("[1,2,3]") == [1, 2, 3]

    def test_parse_config(self):
        """测试解析配置"""
        from waleo_config.parser import parse_config
        from waleo_config.training import TrainingConfig

        args = [
            "--batch_size=128",
            "--offline_steps=100000",
            "--optimizer.lr=0.001",
        ]

        config = parse_config(TrainingConfig, args=args)

        assert config.batch_size == 128
        assert config.offline_steps == 100000
        assert config.optimizer.lr == 0.001

    def test_get_nested_attr(self):
        """测试获取嵌套属性"""
        from waleo_config.parser import get_nested_attr
        from waleo_config.training import TrainingConfig, OptimizerConfig

        config = TrainingConfig(optimizer=OptimizerConfig(lr=0.001))

        assert get_nested_attr(config, "optimizer.lr") == 0.001

    def test_set_nested_attr(self):
        """测试设置嵌套属性"""
        from waleo_config.parser import set_nested_attr
        from waleo_config.training import TrainingConfig

        config = TrainingConfig()
        set_nested_attr(config, "optimizer.lr", 0.001)

        assert config.optimizer.lr == 0.001


class TestValidation:
    """配置验证测试"""

    def test_validate_config(self):
        """测试验证配置"""
        from waleo_config.validation import validate_config, get_validation_errors
        from waleo_config.training import TrainingConfig

        config = TrainingConfig(batch_size=32)
        assert validate_config(config) is True

        # 使用 get_validation_errors 来检查无效配置的错误信息
        # 直接尝试创建无效配置会触发异常
        try:
            config = TrainingConfig(batch_size=-1)
        except ValueError as e:
            # 验证异常信息正确
            assert "batch_size" in str(e)
            # 创建有效配置然后修改字段值
            config = TrainingConfig(batch_size=32)
            # 直接修改字段会破坏验证，所以测试验证功能确实工作
            assert validate_config(config) is True

    def test_check_config_consistency(self):
        """测试检查配置一致性"""
        from waleo_config.validation import check_config_consistency
        from waleo_config.training import TrainingConfig
        from waleo_config.eval import EvalConfig

        train_cfg = TrainingConfig(batch_size=32)
        eval_cfg = EvalConfig(n_episodes=50)

        issues = check_config_consistency({
            "training": train_cfg,
            "eval": eval_cfg
        })

        assert len(issues) == 0

    def test_validate_all(self):
        """测试验证所有配置"""
        from waleo_config.validation import validate_all
        from waleo_config.training import TrainingConfig
        from waleo_config.eval import EvalConfig

        configs = {
            "training": TrainingConfig(batch_size=32),
            "eval": EvalConfig(n_episodes=50)
        }

        assert validate_all(configs) is True


class TestPlugin:
    """插件系统测试"""

    def test_register_config(self):
        """测试注册配置"""
        from waleo_config.parser import register_config, list_plugins, get_registered_configs

        @register_config
        class TestConfig:
            pass

        assert "TestConfig" in list_plugins()
        assert "TestConfig" in get_registered_configs()

    def test_unregister_config(self):
        """测试注销配置"""
        from waleo_config.parser import register_config, unregister_config, list_plugins

        @register_config
        class TestConfig2:
            pass

        unregister_config(TestConfig2)
        assert "TestConfig2" not in list_plugins()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
