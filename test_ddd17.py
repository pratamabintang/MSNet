import torch
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import pickle
import io

# ================= 兼容 numpy 2.x 权重的安全补丁 =================
class PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

class patched_pickle_module:
    Unpickler = PatchedUnpickler

    @staticmethod
    def load(file, **kwargs):
        return PatchedUnpickler(file, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return PatchedUnpickler(io.BytesIO(data), **kwargs).load()
# ===============================================================

# 1. 导入 DDD17 的配置
from configs.config_ddd17 import config

# 2. 导入 DDD17 的数据加载器
try:
    from tools.loader_DDD17 import get_dataloaders
except ImportError:
    print("❌ Error: Could not import 'loader_DDD17'.")
    sys.exit(1)

# 3. 导入 DDD17 原生模型与工具
from network.model_DDD17 import EGHFNet
from tools.utils import RunningConfusionMatrix, colorize_mask


def calculate_pixel_accuracy(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    total_pixels = np.sum(confusion_matrix)
    if total_pixels == 0: return 0.0
    return np.sum(intersection) / total_pixels


def save_single_prediction(logit, filename, save_dir):
    """保存单张预测结果"""
    # 转为 numpy 索引图 [H, W]
    pred = torch.argmax(logit, dim=0).detach().cpu().numpy()

    # 上色
    pred_color = colorize_mask(pred)

    # 构造文件名并保存
    save_path = os.path.join(save_dir, f"{filename}.png")
    cv2.imwrite(save_path, pred_color)


def test():
    parser = argparse.ArgumentParser(description="Test and Save Images for DDD17 Dataset")
    parser.add_argument("--ckpt", type=str, default=None, help="Manually path to checkpoint")
    args = parser.parse_args()

    # --- 1. 确定权重路径 ---
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = config.BEST_MIOU_CKPT_PATH

    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        print("💡 提示：请检查 DDD17 模型权重是否存在，或使用 --ckpt 手动指定。")
        return

    print(f"\n=== 🚀 Testing DDD17 Dataset ===")
    print(f"   Checkpoint: {ckpt_path}")

    # --- 2. 实例化原生模型并加载权重 ---
    model = EGHFNet(num_classes=config.NUM_CLASSES, num_bins=config.NUM_BINS).to(config.DEVICE)

    # ================= 新增：参数量计算功能 =================
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"📦 Model Parameters: {total_params:.2f} M")
    # ========================================================

    print(f"⚡ Loading weights safely...")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False, pickle_module=patched_pickle_module)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k != 'n_averaged'}
    model.load_state_dict(new_state_dict, strict=False)

    # --- 3. 获取测试集数据 ---
    test_loader = get_dataloaders("test")
    test_dataset = test_loader.dataset
    all_file_paths = test_dataset.files  # 包含所有测试集文件的完整路径列表

    print(f"📊 Total Batches: {len(test_loader)}")

    # --- 4. 准备保存总路径 ---
    save_dir = config.RUNS_DIR / "predictions_ddd17"
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Base Output Directory: {save_dir}")

    # --- 5. 初始化指标计算器 ---
    model.eval()
    conf_mat = RunningConfusionMatrix(config.NUM_CLASSES, ignore_label=255)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print("Running Inference...")
    global_idx = 0  # 用于追踪当前图片在整个数据集中的绝对索引

    with torch.no_grad():
        pbar = tqdm(test_loader, ncols=100)
        for i, batch in enumerate(pbar):
            rgb = batch["rgb"].to(config.DEVICE)
            event = batch["event"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)

            # 1. 混合精度推理
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(rgb, event)
                if isinstance(logits, tuple): logits = logits[0]

            # 还原为 float 计算指标
            logits = logits.float()

            # 2. 更新指标
            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

            # 3. 遍历 Batch 保存推理图片 (按序列分类)
            batch_size = logits.shape[0]
            for b in range(batch_size):
                single_logit = logits[b]

                # 利用全局索引获取该图片的原始完整路径
                original_path = all_file_paths[global_idx]
                global_idx += 1

                # 提取序列名称 (如果有相应的上级目录) 和 文件名
                seq_name = original_path.parent.name
                fname = original_path.stem

                # 为当前序列在输出目录下创建专属文件夹
                seq_save_dir = save_dir / seq_name
                os.makedirs(seq_save_dir, exist_ok=True)

                # 将图片保存到对应的序列文件夹中
                save_single_prediction(single_logit, fname, seq_save_dir)

    # --- 6. 计算并输出最终指标 ---
    miou, class_ious = conf_mat.compute()
    pa = calculate_pixel_accuracy(conf_mat.confusion_matrix)

    print(f"\n{'=' * 30}")
    print(f"🏆 Final Results (DDD17)")
    print(f"{'=' * 30}")
    print(f"👉 mIoU: {miou:.4f}")
    print(f"👉 PA:   {pa:.4f}")
    print(f"{'-' * 30}")

    class_names = ['Back', 'Build', 'Fence', 'Person', 'Pole', 'Road']
    for i, iou in enumerate(class_ious):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"   {name:<8}: {iou:.4f}")
    print("-" * 30)
    print(f"✅ All images successfully saved by sequence to: {save_dir}")


if __name__ == "__main__":
    test()