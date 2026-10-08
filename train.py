import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from configs.config import config
from tools.loader import get_dataloaders
from network.model import EGHFNet
from tools.utils import RunningConfusionMatrix
from tools.loss import CrossModalContrastiveLoss  # 引入对比学习损失
import os
import sys
import time
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_iters=1500, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / float(max(1, self.warmup_iters))
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            factor = ((1 - (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power)
            return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]


class OhemCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=True):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weights = torch.tensor([1.0, 1.2, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 3.0, 3.0]).to(config.DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction='none')
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, target):
        loss = self.criterion(pred, target).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)


def smart_init_event_stem(model):
    try:
        rgb_stem = model.backbone.rgb_net.patch_embed1.proj
        evt_stem = model.backbone.evt_net.patch_embed1.proj
        with torch.no_grad():
            rgb_mean = rgb_stem.weight.mean(dim=1, keepdim=True)
            if rgb_mean.shape[0] != evt_stem.weight.shape[0]:
                if rgb_mean.shape[0] > evt_stem.weight.shape[0]:
                    rgb_mean = rgb_mean[:evt_stem.weight.shape[0]]
                else:
                    repeats = (evt_stem.weight.shape[0] // rgb_mean.shape[0]) + 1
                    rgb_mean = rgb_mean.repeat(repeats, 1, 1, 1)[:evt_stem.weight.shape[0]]
            evt_stem.weight.copy_(rgb_mean.repeat(1, evt_stem.in_channels, 1, 1))
            if rgb_stem.bias is not None and evt_stem.bias is not None:
                rgb_bias = rgb_stem.bias
                if rgb_bias.shape[0] > evt_stem.bias.shape[0]:
                    evt_stem.bias.copy_(rgb_bias[:evt_stem.bias.shape[0]])
                else:
                    evt_stem.bias.copy_(rgb_bias)
    except AttributeError:
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
            # 解析模型输出，分离语义分割结果和对比学习特征对
            if isinstance(outputs, tuple):
                logits, contrast_pairs = outputs
            else:
                logits, contrast_pairs = outputs, []

            # 1. 语义分割主损失
            loss_main = criterion(logits, mask)

            # 2. 跨模态对比损失 (CM-SECA)
            loss_ctr = torch.tensor(0.0, device=logits.device)
            if len(contrast_pairs) > 0:
                for r_f, e_f in contrast_pairs:
                    # 使用 AvgPool 下采样特征图以节省显存并加速对比计算
                    r_f_d = F.adaptive_avg_pool2d(r_f, (r_f.shape[2] // 4, r_f.shape[3] // 4))
                    e_f_d = F.adaptive_avg_pool2d(e_f, (e_f.shape[2] // 4, e_f.shape[3] // 4))
                    loss_ctr += ctr_criterion(r_f_d, e_f_d)

            # 融合最终损失
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

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})

    avg_loss = total_loss_meter / len(train_loader)
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


def validate(model, val_loader, epoch, writer, best_miou, optimizer):
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
        }, config.BEST_MIOU_CKPT_PATH)
        return avg_miou, avg_miou
    return avg_miou, best_miou


def main():
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.TXT_LOG_DIR, exist_ok=True)
    sys.stdout = Logger(os.path.join(config.TXT_LOG_DIR, f"log_MUSE_{timestamp}.txt"))

    print(f"=== MUSE-Net (Mamba + Routing + Topology + Contrastive) ===")
    config.LEARNING_RATE = 2e-4
    train_loader = get_dataloaders("train")
    val_loader = get_dataloaders("test")

    model = EGHFNet().to(config.DEVICE)
    smart_init_event_stem(model)

    gdca_params = [p for n, p in model.named_parameters() if "offset_conv" in n and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if "offset_conv" not in n and p.requires_grad]
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': config.LEARNING_RATE},
        {'params': gdca_params, 'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda')
    total_iters = config.EPOCHS * len(train_loader)
    scheduler = WarmupPolyLR(optimizer, max_iters=total_iters, warmup_iters=1500, power=0.9)

    pixels_per_batch = config.BATCH_SIZE * config.IMG_SIZE[0] * config.IMG_SIZE[1]
    min_kept = int(pixels_per_batch * 0.10)
    criterion = OhemCrossEntropyLoss(ignore_index=255, thresh=0.7, min_kept=min_kept, use_weight=True)

    # 初始化对比学习损失
    ctr_criterion = CrossModalContrastiveLoss(temperature=0.1).to(config.DEVICE)

    writer = SummaryWriter(config.LOG_DIR)
    start_epoch = 0
    best_miou = 0.0

    if args.resume and os.path.exists(config.LATEST_CKPT_PATH):
        ckpt = torch.load(config.LATEST_CKPT_PATH, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt['epoch']
        best_miou = ckpt.get('miou', 0.0)

    for epoch in range(start_epoch, config.EPOCHS):
        train_one_epoch(model, train_loader, criterion, ctr_criterion, optimizer, scheduler, epoch, writer, scaler)
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "miou": best_miou,
            "optimizer_state_dict": optimizer.state_dict(),
        }, config.LATEST_CKPT_PATH)

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            _, new_best_miou = validate(model, val_loader, epoch, writer, best_miou, optimizer)
            if new_best_miou > best_miou:
                best_miou = new_best_miou
    writer.close()


if __name__ == "__main__":
    main()
