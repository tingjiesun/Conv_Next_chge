import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent
# 指向包含 ultralytics 包的目录
_local_ultralytics_root = _repo_root / "ultralytics-main" / "ultralytics-main"
if _local_ultralytics_root.exists():
    sys.path.insert(0, str(_local_ultralytics_root))

from ultralytics import YOLO


def on_train_epoch_end(trainer):
    """每轮训练结束后调用"""
    epoch = trainer.epoch + 1
    total_epochs = trainer.epochs

    # 获取当前 epoch 的损失
    loss = trainer.loss.item() if trainer.loss is not None else 0  # 总损失
    box_loss = trainer.loss_items[0].item() if trainer.loss_items is not None else 0  # 边框损失
    cls_loss = trainer.loss_items[1].item() if trainer.loss_items is not None else 0  # 分类损失
    dfl_loss = trainer.loss_items[2].item() if trainer.loss_items is not None else 0  # DFL损失

    print(f"\n{'=' * 50}")
    print(f" Epoch {epoch}/{total_epochs} 训练完成")
    print(f"   总损失: {loss:.4f}")
    print(f"   边框损失 (box): {box_loss:.4f}")
    print(f"   分类损失 (cls): {cls_loss:.4f}")
    print(f"   DFL损失: {dfl_loss:.4f}")


def on_fit_epoch_end(trainer):
    """每轮训练+验证结束后调用（包含验证指标）"""
    epoch = trainer.epoch + 1

    # 获取验证指标 (metrics)
    metrics = trainer.metrics

    if metrics:
        print(f" Epoch {epoch} 验证指标:")

        # 常用指标
        precision = metrics.get('metrics/precision(B)', 0)
        recall = metrics.get('metrics/recall(B)', 0)
        map50 = metrics.get('metrics/mAP50(B)', 0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0)

        print(f"   精确率 (Precision): {precision:.4f}")
        print(f"   召回率 (Recall): {recall:.4f}")
        print(f"   mAP@0.5: {map50:.4f}")
        print(f"   mAP@0.5:0.95: {map50_95:.4f}")

        # 学习率
        lr = trainer.optimizer.param_groups[0]['lr']
        print(f"   当前学习率: {lr:.6f}")


        # 保存到文件（可选）
        with open("training_log.txt", "a") as f:
            f.write(f"{epoch},{precision:.4f},{recall:.4f},{map50:.4f},{map50_95:.4f},{lr:.6f}\n")


def on_train_end(trainer):
    """训练完全结束后调用"""
    print("训练完成！")
    print(f"最佳模型保存在: {trainer.best}")
    print(f"最终模型保存在: {trainer.last}")


# 主训练代码
if __name__ == "__main__":
    # 加载模型
    # 使用正斜杠 / 避免转义问题
    model_cfg = "/tmp/project_convnext/ultralytics-main/ultralytics-main/ultralytics/cfg/models/11/yolo11-convnextv2.yaml"
    model = YOLO(model_cfg)
    print(f"本次训练使用模型: {model_cfg}")

    # 添加自定义回调
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # 开始训练
    model.train(
        data="data.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        device=0,
        project="runs/train",
        name="yolo11_convnext_2",
        workers=0,
        amp=False,
        pretrained=False,  # 关闭预训练
        # --- 针对 ConvNeXt V2 的超参数优化 ---
        optimizer="AdamW",  # 推荐：ConvNeXt 更适合 AdamW
        lr0=0.001,  # 推荐：AdamW 的初始学习率通常较小
        warmup_epochs=5.0,  # 推荐：增加热身轮数以稳定初期训练
        cos_lr=True,  # 可选：使用余弦退火学习率调度器，通常收敛更好

        mosaic=0.3,
        close_mosaic=30,
    )
