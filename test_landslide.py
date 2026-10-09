import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm

from configs.config_landslide import config
from tools.loader_landslide import get_landslide_dataloaders
from network.model_landslide import EGHFNet
from tools.loss import compute_metrics


def save_prediction_vis(rgb_tensor, topo_tensor, mask_tensor, pred_logit, sample_name, save_dir):
    """Saves a multi-panel visual comparison: [RGB | Topo | Ground Truth | Prediction Overlay]."""
    # 1. Un-normalize RGB: [3, H, W] -> [H, W, 3] in [0, 255] BGR
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    rgb_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    rgb_vis = np.clip((rgb_np * std + mean) * 255.0, 0, 255).astype(np.uint8)
    rgb_bgr = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)

    # 2. Topo (channel 0, typically DTM or Slope): [H, W] in [0, 255]
    topo_np = topo_tensor[0].cpu().numpy()
    topo_vis = cv2.applyColorMap(np.clip(topo_np * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # 3. Ground Truth Mask [H, W]
    gt_np = (mask_tensor.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    gt_vis = np.zeros_like(rgb_bgr)
    gt_vis[gt_np == 1] = [0, 255, 0]  # Green for Ground Truth

    # 4. Prediction Mask & Overlay
    pred_prob = torch.sigmoid(pred_logit).squeeze(0).cpu().numpy()
    pred_bin = (pred_prob >= 0.5).astype(np.uint8)
    overlay = rgb_bgr.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[pred_bin == 1] = [0, 0, 255]
    overlay = cv2.addWeighted(overlay, 1.0, red_mask, 0.5, 0.0)

    # Combine into 1 side-by-side row: [RGB | Topo | Ground Truth | Prediction Overlay]
    combined = np.hstack([rgb_bgr, topo_vis, gt_vis, overlay])
    save_path = Path(save_dir) / f"{sample_name}_eval.png"
    cv2.imwrite(str(save_path), combined)


def evaluate_test():
    parser = argparse.ArgumentParser(description="Evaluate MUSE-Net on Landslide Dataset")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--modalities", type=str, default=None, help="Comma-separated modalities, e.g. 'IMAGE,DTM,SLOPE,HILLSHADE'")
    parser.add_argument("--device", type=str, default=None, help="Compute device, e.g. 'cuda:0' or 'cpu'")
    parser.add_argument("--save_vis", action="store_true", default=True, help="Save evaluation overlay visualizations")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory path")
    parser.add_argument("--blacklist_path", "--black_list_path", dest="blacklist_path", type=str, default=None, help="Path to blacklist file")
    parser.add_argument("--output_dir", "--output_training_path", dest="output_dir", type=str, default=None, help="Output directory for predictions")
    parser.add_argument("--val_image_path", "--image_path", dest="image_path", type=str, default=None, help="Direct path to validation IMAGE folder")
    parser.add_argument("--val_mask_path", "--mask_path", dest="mask_path", type=str, default=None, help="Direct path to validation LABEL/MASK folder")
    parser.add_argument("--val_dtm_path", "--dtm_path", dest="dtm_path", type=str, default=None, help="Direct path to validation DTM folder")
    args = parser.parse_args()

    device_str = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if args.data_dir is not None:
        config.DATA_DIR = Path(args.data_dir)
        config.BLACKLIST_PATH = config.DATA_DIR / "black_list.txt"

    if args.blacklist_path is not None:
        config.BLACKLIST_PATH = Path(args.blacklist_path)

    if args.output_dir is not None:
        config.RUNS_DIR = Path(args.output_dir)

    if args.modalities is not None:
        config.MODALITIES = [m.strip().upper() for m in args.modalities.split(",")]
        config.TOPO_INPUT_CHANS = max(1, len([m for m in config.MODALITIES if m != "IMAGE"]))

    ckpt_path = Path(args.ckpt) if args.ckpt else config.BEST_IOU_CKPT_PATH
    if not ckpt_path.exists():
        fallback = config.LATEST_CKPT_PATH
        if fallback.exists():
            ckpt_path = fallback
        else:
            print(f"[Warning] Checkpoint not found at {ckpt_path}. Evaluating untrained model (or specify --ckpt).")

    print(f"\n=== Testing MUSE-Net Landslide Evaluation ===")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Split:      {args.split}")
    print(f"  Device:     {device}")
    print(f"  Modalities: {config.MODALITIES} (Topo: {config.TOPO_INPUT_CHANS} chans)")

    model = EGHFNet(
        topo_in_channels=config.TOPO_INPUT_CHANS,
        num_classes=config.NUM_CLASSES,
        use_graph=config.USE_GRAPH,
    ).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "n_averaged"}
        model.load_state_dict(clean_sd, strict=False)
        recorded_iou = ckpt.get("best_iou", None)
        if recorded_iou is not None:
            print(f"[Loaded] Checkpoint with recorded IoU: {recorded_iou:.4f}")
        else:
            print("[Loaded] Weights loaded successfully.")

    model.eval()

    if ckpt_path.exists():
        if ckpt_path.parent.name == "checkpoints":
            exp_folder = ckpt_path.parent.parent
        else:
            exp_folder = ckpt_path.parent
        vis_dir = exp_folder / f"predictions_vis_{args.split}"
    else:
        vis_dir = config.RUNS_DIR / f"predictions_vis_{args.split}"

    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Visualizations] Saving to: {vis_dir}")

    val_paths = {}
    if args.image_path: val_paths["IMAGE"] = args.image_path
    if args.mask_path: val_paths["LABEL"] = args.mask_path
    if args.dtm_path: val_paths["DTM"] = args.dtm_path

    loader = get_landslide_dataloaders(
        split=args.split,
        batch_size=1,
        modalities=config.MODALITIES,
        num_workers=0,
        data_dir=str(config.DATA_DIR),
        blacklist_path=str(config.BLACKLIST_PATH),
        split_paths=val_paths if val_paths else None,
    )

    total_iou = 0.0
    total_f1 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    count = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating", ncols=100)
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            topo = batch["topo"].to(device)
            mask = batch["mask"].to(device)
            name = batch["name"][0]

            logits = model(rgb, topo)
            metrics = compute_metrics(logits, mask)

            total_iou += metrics["iou"]
            total_f1 += metrics["f1"]
            total_prec += metrics["precision"]
            total_rec += metrics["recall"]
            count += 1

            if args.save_vis:
                save_prediction_vis(
                    rgb_tensor=rgb[0],
                    topo_tensor=topo[0],
                    mask_tensor=mask[0],
                    pred_logit=logits[0],
                    sample_name=name,
                    save_dir=vis_dir,
                )

    mean_iou = total_iou / max(1, count)
    mean_f1 = total_f1 / max(1, count)
    mean_prec = total_prec / max(1, count)
    mean_rec = total_rec / max(1, count)

    print(f"\n==========================================")
    print(f"       Final Evaluation Results ({args.split})")
    print(f"==========================================")
    print(f"  Samples Evaluated: {count}")
    print(f"  mIoU:              {mean_iou * 100:.2f} %")
    print(f"  F1 / Dice Score:   {mean_f1 * 100:.2f} %")
    print(f"  Precision:         {mean_prec * 100:.2f} %")
    print(f"  Recall:            {mean_rec * 100:.2f} %")
    print(f"==========================================\n")


if __name__ == "__main__":
    evaluate_test()
