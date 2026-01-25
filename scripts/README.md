# 脚本集合

开发和部署脚本集合。

## 目录结构

```
scripts/
├── benchmarking/          # 性能分析脚本
│   ├── explain_fps_calculation.py
│   └── profile_training_simple.py
└── training/              # 训练启动脚本
    └── start_h100_training.sh
```

## 性能分析脚本

### explain_fps_calculation.py

详细解释 FPS（每秒帧数）计算方法的示例脚本。

**用途**：
- 教育性脚本，展示如何正确计算仿真 FPS
- 区分步骤 FPS 和环境 FPS
- 考虑向量化环境的影响

**运行**：
```bash
python scripts/benchmarking/explain_fps_calculation.py
```

### profile_training_simple.py

简化的训练性能分析脚本。

**用途**：
- 分析训练循环的性能瓶颈
- 测量各个阶段的耗时
- 识别优化机会

**运行**：
```bash
python scripts/benchmarking/profile_training_simple.py
```

## 训练脚本

### start_h100_training.sh

H100 GPU 训练启动脚本（RJ2506 + PickCube）。

**配置**：
- GPU: GPU 7
- 机器人: rj2506
- 任务: PickCube-v1
- 环境数: 512
- 总步数: 20M

**运行**：
```bash
bash scripts/training/start_h100_training.sh
```

**注意**：
- 修改 `CUDA_VISIBLE_DEVICES` 选择不同 GPU
- 修改脚本参数以适应不同任务
- 日志输出到 `training_h100_gpu7_20M.log`

## 添加新脚本

如果你开发了新的脚本，请：

1. 放在合适的子目录（benchmarking, training, deployment等）
2. 添加执行权限：`chmod +x script.sh`
3. 更新本 README
4. 添加脚本说明和使用示例

## 脚本开发规范

1. **命名**：使用描述性名称（如 `profile_training_simple.py`）
2. **文档**：脚本顶部添加用途说明
3. **参数**：使用 argparse 或环境变量传参
4. **错误处理**：检查前置条件和依赖
5. **日志**：重要步骤打印日志
