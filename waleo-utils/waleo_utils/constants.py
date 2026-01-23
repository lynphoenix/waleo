"""
Waleo 常量定义

包含所有模块使用的常量，包括路径、键名、环境变量等
"""

from pathlib import Path

# ============================================================================
# 观察和动作键名
# ============================================================================
OBS_ENV_STATE = "observation.environment_state"
OBS_STATE = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"
REWARD = "next.reward"

# ============================================================================
# 路径常量
# ============================================================================
WALES_HOME = Path("~/.cache/waleo").expanduser()
WALES_CALIBRATION = WALES_HOME / "calibration"

# 训练相关
CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"

# ============================================================================
# 文件名常量
# ============================================================================
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# ============================================================================
# 通信常量
# ============================================================================
DEFAULT_RPC_PORT = 5000
DEFAULT_RPC_TIMEOUT = 5.0
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

# ============================================================================
# 仿真常量
# ============================================================================
DEFAULT_FPS = 30
DEFAULT_MAX_EPISODE_STEPS = 1000
