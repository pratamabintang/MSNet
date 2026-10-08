import os
from pathlib import Path

class Config:
    def __init__(self):
        self.PROJECT_NAME = "EGHF-Net"
        self.VERSION_SUFFIX = "_DSEC_11cls_RGB-B2_Event-B0"

        self.DSEC_ROOT = Path(r"E:\Datasets\DSEC")
        self.PREPROCESSED_ROOT = self.DSEC_ROOT / "preprocessed_eisnet_aet"
        self.PREPROCESSED_TRAIN = self.PREPROCESSED_ROOT / "train"
        self.PREPROCESSED_TEST = self.PREPROCESSED_ROOT / "test"

        self.PRETRAINED_RGB_PATH = Path("./pretrained/mit_b2.pth")
        self.PRETRAINED_EVT_PATH = Path("./pretrained/mit_b0.pth")

        self.RUNS_DIR = Path("./runs") / (self.PROJECT_NAME + self.VERSION_SUFFIX)
        self.CKPT_DIR = Path("./checkpoints") / self.VERSION_SUFFIX
        self.LOG_DIR = Path("./logs") / self.VERSION_SUFFIX
        self.TXT_LOG_DIR = Path("./txt_logs") / self.VERSION_SUFFIX

        for p in [self.RUNS_DIR, self.CKPT_DIR, self.LOG_DIR, self.TXT_LOG_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        self.LATEST_CKPT_PATH = self.CKPT_DIR / "latest_checkpoint.pth"
        self.BEST_MIOU_CKPT_PATH = self.CKPT_DIR / "best_miou_checkpoint.pth"

        self.RGB_BACKBONE = 'b2'
        self.EVENT_BACKBONE = 'b0'
        self.NUM_CLASSES = 11
        self.NUM_BINS = 5
        self.EVENT_INPUT_CHANS = self.NUM_BINS * 2
        self.IMG_SIZE = (440, 640)
        self.TRAIN_CROP_SIZE = (440, 640)

        self.DEVICE = "cuda:0"
        self.EPOCHS = 100
        self.BATCH_SIZE = 8
        self.ACCUMULATION_STEPS = 4
        self.LEARNING_RATE = 2e-4
        self.WEIGHT_DECAY = 0.01
        self.VAL_INTERVAL = 1

        self.LOSS_WEIGHTS = {'lovasz': 0.75, 'ce': 1.0, 'boundary': 0.5, 'detail': 1.0}
        self.OHEM_KEEP_RATIO = 0.20
        self.ENABLE_COPY_PASTE = True

config = Config()