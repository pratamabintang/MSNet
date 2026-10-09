import os
from pathlib import Path
import torch

class Config:
    def __init__(self):
        self.PROJECT_NAME = "MUSE-Net-Landslide"
        self.VERSION_SUFFIX = "_RGB-B2_Topo-B0_512x512"

        # Dataset directories and blacklist
        self.DATA_DIR = Path("datasets/landslide")
        self.BLACKLIST_PATH = self.DATA_DIR / "black_list.txt"

        # Modalities: optical RGB ("IMAGE") + LiDAR topography rasters
        # Supported: "IMAGE", "DTM", "SLOPE", "ASPECT", "ASPECT_COS", "ASPECT_SIN", "HILLSHADE"
        self.MODALITIES = ["IMAGE", "DTM", "SLOPE", "HILLSHADE"]
        self.RGB_CHANS = 3
        self.TOPO_INPUT_CHANS = max(1, len([m for m in self.MODALITIES if m != "IMAGE"]))

        # Pretrained SegFormer weights
        drive_rgb = Path("/content/drive/MyDrive/project/MSNet/pretrained/b2")
        drive_topo = Path("/content/drive/MyDrive/project/MSNet/pretrained/b0")
        self.PRETRAINED_RGB_PATH = drive_rgb if drive_rgb.exists() else Path("./pretrained/mit_b2.pth")
        self.PRETRAINED_TOPO_PATH = drive_topo if drive_topo.exists() else Path("./pretrained/mit_b0.pth")

        # Output and artifact directories (all artifacts grouped in one subfolder)
        self.RUNS_DIR = Path("./runs")
        self.EXP_DIR = self.RUNS_DIR / (self.PROJECT_NAME + self.VERSION_SUFFIX)
        self.CKPT_DIR = self.EXP_DIR / "checkpoints"
        self.LOG_DIR = self.EXP_DIR / "logs"
        self.TXT_LOG_DIR = self.EXP_DIR / "txt_logs"

        for p in [self.EXP_DIR, self.CKPT_DIR, self.LOG_DIR, self.TXT_LOG_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        self.LATEST_CKPT_PATH = self.CKPT_DIR / "latest_checkpoint.pth"
        self.BEST_IOU_CKPT_PATH = self.CKPT_DIR / "best_iou_checkpoint.pth"

        # Architecture choices
        self.RGB_BACKBONE = "b2"
        self.TOPO_BACKBONE = "b0"
        self.NUM_CLASSES = 1       # Binary landslide mask (0: Background, 1: Landslide)
        self.IMG_SIZE = 512
        self.USE_GRAPH = True      # Enable SDGSR (LocalTopoGraphConv)

        # Training hyperparameters
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.EPOCHS = 50
        self.BATCH_SIZE = 4
        self.ACCUMULATION_STEPS = 1
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-4

        # Evaluation and Checkpoint Intervals
        self.EVAL_INTERVAL = 1     # Evaluate every 1 epoch
        self.SAVE_INTERVAL = 5     # Save persistent model checkpoint every 5 epochs

        # Mixed Precision (AMP)
        self.AMP = True
        self.AMP_DTYPE = "auto"    # "auto" (bfloat16 if supported else float16), "fp16", or "bf16"
        self.GRAD_CLIP = 1.0

        # Loss weights
        self.STRUCTURE_LOSS_WEIGHT = 1.0
        self.CONTRASTIVE_LOSS_WEIGHT = 0.1  # CM-SECA at Stage 4

        # Staging & optimization
        self.SKIP_SIZE_CHECK = True

        # Run mode for ablation: 'real' (100% Topo), 'zero' (0% Topo baseline), or 'all'
        self.RUN_MODE = "real"

config = Config()
