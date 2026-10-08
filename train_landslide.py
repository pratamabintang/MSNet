import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.config_landslide import config
from tools.loader_landslide import get_landslide_dataloaders
from network.model_landslide import EGHFNet
from tools.loss import (
    structure_loss,
    CrossModalContrastiveLoss,
    compute_metrics,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class Logger:
    """Tee stdout to both terminal and a persistent log file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train MUSE-Net for Landslide Segmentation (RGB + LiDAR Topography)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--modalities", type=str, default=None, help="Comma-separated modalities, e.g. 'IMAGE,DTM,SLOPE,HILLSHADE'")
    parser.add_argument("--run_mode", type=str, default="real", help="'real' (100% Topo), 'zero'/'zeroo' (0% Topo ablation), 'all'")
    parser.add_argument("--use_graph", type=lambda x: str(x).lower() == "true", default=True, help="Enable/disable SDGSR graph convolution")
    parser.add_argument("--eval_interval", type=int, default=1, help="Run evaluation every N epochs (default: 1)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save persistent checkpoint every N epochs (default: 5)")
    parser.add_argument("--amp", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Enable/disable PyTorch AMP")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "fp16", "float16", "bf16", "bfloat16"], help="AMP precision data type")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--skip_shape_check", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Skip raster shape checking")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pth")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--device", type=str, default=None, help="Compute device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, writer, scaler, device, ctr_criterion, amp_enabled, amp_dtype, grad_clip):
    model.train()
    total_loss = 0.0
    accum_steps = config.ACCUMULATION_STEPS
    use_cuda = device.type == "cuda"
    do_amp = amp_enabled and use_cuda

    pbar = tqdm(train_loader, desc=f"Train Ep {epoch + 1}/{config.EPOCHS}", ncols=120)
    optimizer.zero_grad()

    for i, batch in enumerate(pbar):
        rgb = batch["rgb"].to(device, non_blocking=True)
        topo = batch["topo"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=do_amp):
            outputs = model(rgb, topo)
            logits, contrast_pairs = outputs if isinstance(outputs, tuple) else (outputs, [])
            loss_main = structure_loss(logits, mask)

            loss_ctr = torch.tensor(0.0, device=device)
            if len(contrast_pairs) > 0:
                for r_f, t_f in contrast_pairs:
                    r_f_d = F.adaptive_avg_pool2d(r_f, (r_f.shape[2] // 4, r_f.shape[3] // 4))
                    t_f_d = F.adaptive_avg_pool2d(t_f, (t_f.shape[2] // 4, t_f.shape[3] // 4))
                    loss_ctr += ctr_criterion(r_f_d, t_f_d)

            loss = config.STRUCTURE_LOSS_WEIGHT * loss_main + config.CONTRASTIVE_LOSS_WEIGHT * loss_ctr
            loss_norm = loss / accum_steps

        if scaler.is_enabled():
            scaler.scale(loss_norm).backward()
        else:
            loss_norm.backward()

        total_loss += loss.item()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            optimizer.zero_grad()

        curr_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{curr_lr:.2e}"})

    scheduler.step()
    avg_loss = total_loss / max(1, len(train_loader))
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    return avg_loss


def validate(model, val_loader, epoch, writer, device, amp_enabled, amp_dtype):
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    count = 0
    do_amp = amp_enabled and device.type == "cuda"

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val Ep {epoch + 1}", ncols=100)
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            topo = batch["topo"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=do_amp):
                logits = model(rgb, topo)
                loss = structure_loss(logits, mask)

            metrics = compute_metrics(logits, mask)

            total_loss += loss.item()
            total_iou += metrics["iou"]
            total_f1 += metrics["f1"]
            total_prec += metrics["precision"]
            total_rec += metrics["recall"]
            count += 1

    avg_loss = total_loss / max(1, count)
    avg_iou = total_iou / max(1, count)
    avg_f1 = total_f1 / max(1, count)
    avg_prec = total_prec / max(1, count)
    avg_rec = total_rec / max(1, count)

    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/IoU", avg_iou, epoch)
    writer.add_scalar("Val/F1", avg_f1, epoch)

    print(f"\n[Validation Epoch {epoch + 1}] Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | F1/Dice: {avg_f1:.4f} | Prec: {avg_prec:.4f} | Rec: {avg_rec:.4f}")
    return avg_iou, avg_f1, avg_loss


def run_training(run_mode="real"):
    args = parse_args()

    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.data_dir is not None:
        config.DATA_DIR = Path(args.data_dir)
        config.BLACKLIST_PATH = config.DATA_DIR / "black_list.txt"
    if args.modalities is not None:
        config.MODALITIES = [m.strip().upper() for m in args.modalities.split(",")]
        config.TOPO_INPUT_CHANS = max(1, len([m for m in config.MODALITIES if m != "IMAGE"]))
    if args.use_graph is not None:
        config.USE_GRAPH = args.use_graph

    # Normalize mode flag (handles 'zeroo' alias as 'zero')
    raw_mode = (run_mode or args.run_mode).lower()
    if raw_mode in ("zero", "zeroo", "0", "0%", "dtm0", "dtm_0", "dtm_0%"):
        mode = "zero"
    elif raw_mode in ("real", "100", "100%", "dtm100", "dtm_100", "dtm_100%"):
        mode = "real"
    else:
        mode = "real"

    config.RUN_MODE = mode
    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Resolve AMP dtype (matches reference repository logic)
    if str(args.amp_dtype).lower() in ("bf16", "bfloat16"):
        target_amp_dtype = torch.bfloat16
    elif str(args.amp_dtype).lower() in ("fp16", "float16"):
        target_amp_dtype = torch.float16
    else:
        # "auto": prefer bfloat16 if GPU natively supports it, else float16
        target_amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    # Set up logging directories
    run_suffix = f"_{mode.upper()}_Topo{config.TOPO_INPUT_CHANS}ch"
    log_dir = config.LOG_DIR.parent / (config.PROJECT_NAME + run_suffix)
    txt_log_dir = config.TXT_LOG_DIR.parent / (config.PROJECT_NAME + run_suffix)
    ckpt_dir = config.CKPT_DIR.parent / (config.PROJECT_NAME + run_suffix)

    for p in [log_dir, txt_log_dir, ckpt_dir]:
        p.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sys.stdout = Logger(txt_log_dir / f"train_{timestamp}.log")
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"\n=======================================================")
    print(f"  MUSE-Net: Cross-Modal Landslide Semantic Segmentation")
    print(f"=======================================================")
    print(f"  Device:           {device}")
    print(f"  Run Mode:         {mode.upper()} ({'0% Topo Ablation' if mode == 'zero' else '100% Real Topo'})")
    print(f"  Modalities:       {config.MODALITIES}")
    print(f"  LiDAR Topo Chans: {config.TOPO_INPUT_CHANS}")
    print(f"  SDGSR Graph:      {config.USE_GRAPH}")
    print(f"  AMP Enabled:      {args.amp} (Dtype: {target_amp_dtype})")
    print(f"  Eval Interval:    Setiap {args.eval_interval} epoch")
    print(f"  Save Interval:    Setiap {args.save_interval} epoch")
    print(f"  Skip Shape Check: {args.skip_shape_check}")
    print(f"  Image Size:       {config.IMG_SIZE} x {config.IMG_SIZE}")
    print(f"  Batch Size:       {config.BATCH_SIZE}")
    print(f"  Learning Rate:    {config.LEARNING_RATE}")
    print(f"  Epochs:           {config.EPOCHS}")
    print(f"  Checkpoints:      {ckpt_dir}")
    print(f"=======================================================\n")

    # Dataloaders
    train_loader = get_landslide_dataloaders(
        "train",
        batch_size=config.BATCH_SIZE,
        modalities=config.MODALITIES,
        run_mode=mode,
        num_workers=args.num_workers,
        skip_shape_check=args.skip_shape_check,
    )
    val_loader = get_landslide_dataloaders(
        "val",
        batch_size=config.BATCH_SIZE,
        modalities=config.MODALITIES,
        run_mode=mode,
        num_workers=args.num_workers,
        skip_shape_check=args.skip_shape_check,
    )

    # Instantiate Model
    model = EGHFNet(
        topo_in_channels=config.TOPO_INPUT_CHANS,
        num_classes=config.NUM_CLASSES,
        use_graph=config.USE_GRAPH,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model Parameters: {total_params:.2f} M")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # GradScaler is utilized when AMP is active on CUDA with float16
    use_scaler = args.amp and (device.type == "cuda") and (target_amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ctr_criterion = CrossModalContrastiveLoss(temperature=0.1)

    start_epoch = 0
    best_iou = 0.0

    latest_ckpt_path = ckpt_dir / "latest_checkpoint.pth"
    best_ckpt_path = ckpt_dir / "best_iou_checkpoint.pth"

    if args.resume and latest_ckpt_path.exists():
        print(f"Resuming from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou = ckpt.get("best_iou", 0.0)
        print(f"Resumed at epoch {start_epoch + 1} with Best IoU: {best_iou:.4f}")

    for epoch in range(start_epoch, config.EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            writer=writer,
            scaler=scaler,
            device=device,
            ctr_criterion=ctr_criterion,
            amp_enabled=args.amp,
            amp_dtype=target_amp_dtype,
            grad_clip=args.grad_clip,
        )

        # 1. Evaluasi dijalankan setiap interval (default: setiap 1 epoch)
        if (epoch + 1) % args.eval_interval == 0:
            val_iou, val_f1, val_loss = validate(
                model=model,
                val_loader=val_loader,
                epoch=epoch,
                writer=writer,
                device=device,
                amp_enabled=args.amp,
                amp_dtype=target_amp_dtype,
            )

            is_best = val_iou > best_iou
            if is_best:
                best_iou = val_iou
                print(f"[Best IoU] New Best IoU: {best_iou:.4f}! Saving to {best_ckpt_path.name}")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_iou": best_iou,
                    "val_f1": val_f1,
                    "modalities": config.MODALITIES,
                }, best_ckpt_path)

        # 2. Simpan model berkala setiap interval (default: setiap 5 epoch)
        if (epoch + 1) % args.save_interval == 0:
            interval_ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            print(f"[Checkpoint Interval] Saving epoch {epoch + 1} checkpoint to {interval_ckpt_path.name}...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_iou": best_iou,
                "modalities": config.MODALITIES,
            }, interval_ckpt_path)

        # 3. Selalu perbarui latest checkpoint untuk resume
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "best_iou": best_iou,
            "modalities": config.MODALITIES,
        }, latest_ckpt_path)

    print(f"\n[Done] Training complete ({mode.upper()})! Best Validation IoU: {best_iou:.4f}")
    writer.close()
    return best_iou


def main():
    args = parse_args()
    raw_mode = args.run_mode.lower()

    if raw_mode in ("all", "both"):
        print(">>> Memulai Eksperimen Ablasi Penuh: RUN_MODE = all (DTM 0% vs DTM 100%) <<<")
        print("\n--- [Bagian 1/2] Training Baseline Kontrol: Topo 0% ('zero') ---")
        best_zero = run_training(run_mode="zero")
        print("\n--- [Bagian 2/2] Training Model Usulan: Topo 100% ('real') ---")
        best_real = run_training(run_mode="real")
        print("\n=======================================================")
        print("            RINGKASAN HASIL ABLASI (ALL):              ")
        print("=======================================================")
        print(f"  Best Val IoU Topo 0%   (Baseline) : {best_zero * 100:.2f} %")
        print(f"  Best Val IoU Topo 100% (MUSE-Net) : {best_real * 100:.2f} %")
        diff = (best_real - best_zero) * 100
        print(f"  Peningkatan Kontribusi LiDAR Topo : {diff:+.2f} %")
        print("=======================================================\n")
    else:
        run_training(run_mode=args.run_mode)


if __name__ == "__main__":
    main()
