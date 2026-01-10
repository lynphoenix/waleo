# ManiSkill3 PickCube 训练示例 - 完整实现

## 📋 概述

这是一个完整的 ManiSkill3 训练示例，展示了如何使用 waleo-sim 模块训练机器人抓取策略。

### 核心特性

- ✅ **ManiSkill3**: 最新版本 (3.0.0b22)
- ✅ **Franka Panda 机械臂**：7-DOF 机器人
- ✅ **双相机视觉**：头部相机 + 腕部相机
- ✅ **PickCube 任务**：抓取立方体
- ✅ **端到端训练**：从原始像素到动作
- ✅ **完整流程**：数据采集 → 训练 → 评估

## 📁 文件结构

```
waleo-sim/
├── waleo_sim/
│   ├── tasks/
│   │   └── maniskill_pick_cube.py    # ManiSkill 环境封装
│   └── backends/
│       └── maniskill.py               # ManiSkill 后端
└── examples/maniskill/
    ├── __init__.py                     # 模块初始化
    ├── policy.py                        # 策略网络定义
    ├── collect_data.py                  # 数据采集脚本
    ├── train.py                         # 训练脚本
    ├── evaluate.py                      # 评估脚本
    ├── quickstart.py                    # 快速开始
    └── README.md                        # 使用说明
```

## 🚀 快速开始

### 方法 1：使用快速开始脚本（推荐）

```bash
cd examples/maniskill
python quickstart.py
```

这会自动：
1. 测试环境是否正常工作
2. 创建一个最小化数据集
3. 测试训练流程

### 方法 2：手动执行完整流程

#### 步骤 1：数据采集

```bash
cd examples/maniskill
python collect_data.py
```

**说明：**
- 使用启发式策略采集 50 个 episode
- 数据保存在 `./data/demonstrations/dataset_final.npz`

#### 步骤 2：训练策略

```bash
python train.py --data ./data/demonstrations/dataset_final.npz
```

**参数：**
- `--epochs 100`：训练轮数
- `--batch-size 32`：批大小
- `--lr 1e-4`：学习率
- `--device cuda`：训练设备

#### 步骤 3：评估策略

```bash
python evaluate.py --checkpoint ./checkpoints/best_checkpoint.pt
```

## 🧠 网络架构

```
输入层：
├── 头部相机 RGB → CNN → Feature (256D)
├── 腕部相机 RGB → CNN → Feature (256D)
└── 机器人状态 → Direct (9D)

融合层：
└── Concat(512D) → MLP(256D) → MLP(256D) → Action(7D)

输出层：
└── 7维动作（位置 + 姿态 + 夹爪）
```

## 📊 训练流程

### 数据采集

```python
# 使用启发式策略采集演示
collector = DataCollector(env)
dataset = collector.collect_dataset(
    num_episodes=50,
    max_steps=200,
)
```

### 行为克隆训练

```python
# 创建策略网络
policy = VisualPolicy(
    image_size=(224, 224),
    state_dim=9,
    action_dim=7,
)

# 训练
trainer = PolicyTrainer(policy, lr=1e-4)
for epoch in range(100):
    for batch in dataloader:
        loss = trainer.train_step(batch)
```

### 评估

```python
# 评估策略
results = evaluate(
    checkpoint_path="best_checkpoint.pt",
    num_episodes=50,
)

print(f"Success Rate: {results['success_rate']}%")
```

## 📈 预期结果

| 阶段 | 指标 | 说明 |
|------|------|------|
| 数据采集 | 成功率 10-30% | 启发式策略的表现 |
| 训练 | 损失下降 | MSE loss 随 epoch 减少 |
| 评估 | 成功率 20-50% | 取决于数据质量和训练轮数 |

## ⚙️ 环境要求

### 必需

```bash
# ManiSkill3 (最新版本)
pip install mani_skill

# PyTorch
pip install torch torchvision

# 其他依赖
pip install numpy tqdm gymnasium
```

### 可选（用于渲染）

```bash
# OpenCV（如果需要可视化）
pip install opencv-python
```

## 🔧 自定义配置

### 调整图像尺寸

```python
env = ManiSkillPickCubeEnv(
    image_size=(128, 128),  # 降低分辨率
)
```

### 关闭相机

```python
env = ManiSkillPickCubeEnv(
    use_cameras=False,  # 只使用状态
)
```

### 调整网络大小

```python
policy = VisualPolicy(
    hidden_dim=512,  # 增大隐藏层
    feature_dim=512,  # 增大特征维度
)
```

## 📝 代码示例

### 使用环境

```python
from waleo_sim.tasks.maniskill_pick_cube import ManiSkillPickCubeEnv

# 创建环境
env = ManiSkillPickCubeEnv(
    robot_type="panda",
    image_size=(224, 224),
    use_cameras=True,
)

# 重置
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# 渲染
frame = env.render(mode="rgb_array")

# 关闭
env.close()
```

### 使用策略

```python
from examples.maniskill.policy import VisualPolicy

# 创建策略
policy = VisualPolicy(
    image_size=(224, 224),
    state_dim=9,
    action_dim=7,
)

# 选择动作
action = policy.select_action(observations)
```

## 🐛 故障排除

### 问题：ImportError: No module named 'mani_skill2'

**解决：**
```bash
pip install mani_skill2
```

### 问题：CUDA out of memory

**解决：**
- 减小 batch size
- 减小图像尺寸
- 使用 CPU 训练

### 问题：策略不收敛

**解决：**
- 检查数据质量
- 调整学习率
- 增加训练轮数
- 采集更多数据

## 📚 参考资料

- [ManiSkill2 文档](https://maniskill.readthedocs.io/)
- [ManiSkill2 GitHub](https://github.com/haosulab/ManiSkill2)
- [Franka Panda 机器人](https://www.franka.de/)

## 🎯 下一步

1. **改进数据采集**
   - 使用更好的专家策略
   - 采集更多样化的数据
   - 添加数据增强

2. **改进网络**
   - 使用更强大的 CNN（ResNet）
   - 添加 Transformer
   - 使用注意力机制

3. **改进训练**
   - 使用强化学习微调
   - 添加课程学习
   - 使用域随机化

4. **扩展到其他任务**
   - PushCube（推立方体）
   - StackCube（堆叠立方体）
   - 其他 ManiSkill 任务
