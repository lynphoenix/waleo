#!/bin/bash
# 自动评估训练中的检查点

TRAIN_LOG="training_log_visual_10M_fast.txt"
CHECKPOINT_DIR="checkpoints"
EVAL_INTERVAL=200000  # 每 20 万步评估一次
LAST_EVAL_STEP=0

echo "=========================================="
echo "自动评估脚本已启动"
echo "=========================================="
echo "训练日志: $TRAIN_LOG"
echo "检查点目录: $CHECKPOINT_DIR"
echo "评估间隔: $EVAL_INTERVAL 步"
echo "=========================================="
echo ""

while true; do
    # 获取当前训练步数
    if [ -f "$TRAIN_LOG" ]; then
        CURRENT_STEP=$(grep "Step " "$TRAIN_LOG" | tail -1 | grep -oP "Step \K[0-9]+" || echo "0")

        # 检查是否需要评估
        if [ "$CURRENT_STEP" -ge "$((LAST_EVAL_STEP + EVAL_INTERVAL))" ]; then
            EVAL_STEP=$(( (CURRENT_STEP / EVAL_INTERVAL) * EVAL_INTERVAL ))

            # 查找对应的检查点
            CHECKPOINT_FILE=$(find "$CHECKPOINT_DIR" -name "ppo_visual_update_${EVAL_STEP}.pt" 2>/dev/null | head -1)

            if [ -n "$CHECKPOINT_FILE" ]; then
                echo ""
                echo "=========================================="
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估: Step ${EVAL_STEP}"
                echo "检查点: $CHECKPOINT_FILE"
                echo "=========================================="

                # 运行评估
                CUDA_VISIBLE_DEVICES=0 conda run -n waleo python examples/maniskill/evaluate_ppo_visual.py \
                    --checkpoint "$CHECKPOINT_FILE" \
                    --episodes 20 \
                    --image-size 64

                LAST_EVAL_STEP=$EVAL_STEP

                echo ""
                echo "评估完成，继续监控训练..."
            fi
        fi
    fi

    # 等待 60 秒后再次检查
    sleep 60
done
