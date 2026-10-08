import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys
import time
import argparse
import random
import numpy as np

from configs.config_ddd17 import config
from tools.loader_DDD17 import get_dataloaders
from network.model_DDD17 import EGHFNet
from tools.utils import RunningConfusionMatrix
from tools.loss import CrossModalContrastiveLoss

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1, min_lr=0.0):
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (
                1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [max(group["lr"] * decay_factor, self.min_lr) for group in self.optimizer.param_groups]


def ensure_backbone_weights(model):
    path_rgb = config.PRETRAINED_RGB_PATH
    path_evt = config.PRETRAINED_EVT_PATH
    if os.path.exists(path_rgb):
        try:
            model.backbone.rgb_net.load_state_dict(torch.load(path_rgb, map_location='cpu'), strict=False)
        except Exception:
            pass
    if os.path.exists(path_evt):
        try:
            model.backbone.evt_net.load_state_dict(torch.load(path_evt, map_location='cpu', weights_only=False),
                                                   strict=False)
        except Exception:
            pass


def train_one_epoch(model, train_loader, criterion, ctr_criterion, optimizer, scheduler, epoch, writer, scaler):
    model.train()
    total_loss_meter = 0.0
    accum_steps = config.ACCUMULATION_STEPS

    pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{config.EPOCHS}", ncols=120)
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        rgb = batch["rgb"].to(config.DEVICE)
        event = batch["event"].to(config.DEVICE)
        mask = batch["mask"].to(config.DEVICE)

        with torch.amp.autocast('cuda'):
            outputs = model(rgb, event)
            if isinstance(outputs, tuple):
                logits, contrast_pairs = outputs
            else:
                logits, contrast_pairs = outputs, []

            loss_main = criterion(logits, mask)

            # CM-SECA 损失计算
            loss_ctr = torch.tensor(0.0, device=logits.device)
            if len(contrast_pairs) > 0:
                for r_f, e_f in contrast_pairs:
                    r_f_d = F.adaptive_avg_pool2d(r_f, (r_f.shape[2] // 4, r_f.shape[3] // 4))
                    e_f_d = F.adaptive_avg_pool2d(e_f, (e_f.shape[2] // 4, e_f.shape[3] // 4))
                    loss_ctr += ctr_criterion(r_f_d, e_f_d)

            loss = loss_main + 0.1 * loss_ctr
            loss_norm = loss / accum_steps

        scaler.scale(loss_norm).backward()
        total_loss_meter += loss.item()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        pbar.set_postfix({"L": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})

    avg_loss = total_loss_meter / len(train_loader)
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


# 注意：这里增加了 scheduler 和 scaler 作为参数，确保最佳模型也可以完美断点续传
def validate(model, val_loader, epoch, writer, best_miou, optimizer, scheduler, scaler):
    model.eval()
    torch.cuda.empty_cache()
    conf_mat = RunningConfusionMatrix(config.NUM_CLASSES, ignore_label=255)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val Ep {epoch + 1}", ncols=100)
        for batch in pbar:
            rgb = batch["rgb"].to(config.DEVICE)
            event = batch["event"].to(config.DEVICE)
            mask = batch["mask"].to(config.DEVICE)
            logits = model(rgb, event)
            if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(logits, dim=1)
            conf_mat.update(preds, mask)

    avg_miou, class_ious = conf_mat.compute()
    writer.add_scalar("Val/MIoU", avg_miou, epoch)
    print(f"\n>>> Global mIoU: {avg_miou:.4f} (Best: {max(best_miou, avg_miou):.4f}) <<<")

    if avg_miou > best_miou:
        print(f"🔥 Best model updated (mIoU: {avg_miou:.4f})")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": avg_miou,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # 新增
            "scaler_state_dict": scaler.state_dict(),  # 新增
        }, config.BEST_MIOU_CKPT_PATH)
        return avg_miou, avg_miou
    return avg_miou, best_miou


def main():
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.TXT_LOG_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(config.TXT_LOG_DIR, f"log_DDD17_{timestamp}.txt"))

    model = EGHFNet(num_classes=config.NUM_CLASSES, num_bins=config.NUM_BINS).to(config.DEVICE)
    if not args.resume: ensure_backbone_weights(model)

    train_loader = get_dataloaders("train")
    val_loader = get_dataloaders("test")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LEARNING_RATE},
        {'params': head_params, 'lr': config.LEARNING_RATE * 10.0}
    ], lr=config.LEARNING_RATE)

    scaler = torch.amp.GradScaler('cuda')
    total_iters = config.EPOCHS * len(train_loader)
    scheduler = PolynomialLR(optimizer, total_iters=total_iters, power=1.0, min_lr=1e-5)

    criterion = nn.CrossEntropyLoss(ignore_index=255).to(config.DEVICE)
    ctr_criterion = CrossModalContrastiveLoss(temperature=0.1).to(config.DEVICE)

    writer = SummaryWriter(config.LOG_DIR)

    # 初始化变量，防止未触发 resume 时报错
    start_epoch = 0
    best_miou = 0.0

    # 完整读取 Checkpoint 逻辑
    if args.resume and os.path.exists(config.LATEST_CKPT_PATH):
        print(f"==> Loading checkpoint from {config.LATEST_CKPT_PATH}")
        ckpt = torch.load(config.LATEST_CKPT_PATH, map_location=config.DEVICE, weights_only=False)

        # 恢复模型权重
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        # 恢复训练进度与评估指标
        start_epoch = ckpt.get('epoch', 0)
        best_miou = ckpt.get('miou', 0.0)

        # 恢复优化器与调度器状态
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])

        print(f"==> Resumed successfully at epoch {start_epoch} (Best mIoU: {best_miou:.4f})")
    elif args.resume:
        print(
            f"==> Warning: Resume flag set, but checkpoint not found at {config.LATEST_CKPT_PATH}. Starting from scratch.")

    # 训练主循环：从 start_epoch 开始
    for epoch in range(start_epoch, config.EPOCHS):
        train_one_epoch(model, train_loader, criterion, ctr_criterion, optimizer, scheduler, epoch, writer, scaler)

        # 完整保存 最新 Checkpoint 逻辑
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": best_miou,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # 新增
            "scaler_state_dict": scaler.state_dict(),  # 新增
        }, config.LATEST_CKPT_PATH)

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            # 传递 scheduler 和 scaler 给 validate 函数
            _, new_best_miou = validate(model, val_loader, epoch, writer, best_miou, optimizer, scheduler, scaler)
            if new_best_miou > best_miou: best_miou = new_best_miou

    writer.close()


if __name__ == "__main__":
    main()