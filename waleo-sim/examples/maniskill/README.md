# ManiSkill3 PickCube 训练示例

这是一个完整的 ManiSkill3 训练示例，使用 Franka Panda 机械臂执行抓取立方体任务。

## 特点

- **机器人**: Franka Panda 7-DOF 机械臂
- **任务**: PickCube（抓取立方体）
- **传感器**: 双相机（头部相机 + 腕部相机）
- **控制**: 7维关节位置控制
- **策略**: 基于视觉的神经网络策略

## 环境要求

```bash
# 安装 ManiSkill3
pip install mani_skill

# 安装其他依赖
pip install torch torchvision numpy tqdm gymnasium
```

## 快速开始

### 1. 数据采集

首先采集演示数据：

```bash
python collect_data.py
```

这会：
- 创建 ManiSkill PickCube 环境
- 使用启发式策略采集 50 个 episode
- 保存数据到 `./data/demonstrations/`

**参数调整：**
- 修改 `num_episodes` 增加数据量
- 修改 `max_steps` 调整 episode 长度
- 修改 `save_dir` 改变保存位置

### 2. 训练策略

使用采集的数据训练策略：

```bash
python train.py
```

**自定义训练：**
```bash
python train.py --data ./data/demonstrations/dataset_final.npz \
                   --epochs 100 \
                   --batch-size 32 \
                   --lr 1e-4 \
                   --save-dir ./checkpoints
```

**参数说明：**
- `--data`: 数据文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批大小
- `--lr`: 学习率
- `--save-dir`: 检查点保存目录
- `--device`: 训练设备（cuda/cpu）

### 3. 评估策略

评估训练好的策略：

```bash
python evaluate.py --checkpoint ./checkpoints/best_checkpoint.pt
```

**评估参数：**
```bash
python evaluate.py --checkpoint ./checkpoints/best_checkpoint.pt \
                    --episodes 50 \
                    --max-steps 200 \
                    --render
```

**交互式演示：**
```bash
python evaluate.py --checkpoint ./checkpoints/best_checkpoint.pt --interactive
```

## 文件结构

```
examples/maniskill/
├── policy.py              # 策略网络定义
├── collect_data.py        # 数据采集脚本
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
└── README.md              # 本文件
```

## 网络架构

```
输入:
- 头部相机 RGB: (3, 224, 224)
- 腕部相机 RGB: (3, 224, 224)
- 机器人状态: (9,)

处理:
├── Head Camera → CNN → Feature (256)
├── Wrist Camera → CNN → Feature (256)
└── Robot State → Direct (9)

融合 → MLP → Action (7)
```

## 训练流程

1. **数据采集** → 使用启发式策略收集演示
2. **行为克隆** → 训练神经网络模仿演示
3. **评估** → 测试策略在真实环境中的表现

## 预期结果

- **数据采集**: 成功率约 10-30%（取决于启发式策略质量）
- **训练**: MSE 损失随时间下降
- **评估**: 训练后成功率应该有所提高

## 注意事项

1. **计算资源**: 训练需要 GPU，推荐使用 CUDA
2. **训练时间**: 50 个 episode × 100 epochs 大约需要 1-2 小时（取决于硬件）
3. **数据质量**: 启发式策略的质量直接影响最终策略性能
4. **超参数调整**: 如果效果不好，可以调整学习率、batch size 等

## 故障排除

### ImportError: No module named 'mani_skill2'
```bash
pip install mani_skill2
```

### CUDA out of memory
- 减小 `batch_size`
- 减小图像尺寸
- 使用 CPU 训练（`--device cpu`）

### 策略不收敛
- 检查数据质量（是否成功采集到数据）
- 调整学习率
- 增加训练轮数
- 采集更多数据

## 扩展方向

1. **改进数据采集**:
   - 使用更好的专家策略
   - 采集更多样化的数据
   - 数据增强（图像变换等）

2. **改进网络架构**:
   - 使用更强大的 CNN（ResNet, EfficientNet）
   - 添加 Transformer 层
   - 使用更复杂的融合策略

3. **改进训练方法**:
   - 使用强化学习微调
   - 添加数据增强
   - 使用课程学习

## 参考资料

- [ManiSkill2 文档](https://maniskill.readthedocs.io/)
- [ManiSkill2 GitHub](https://github.com/haosulab/ManiSkill2)
- [Franka Panda 机器人](https://www.franka.de/)
