import os
from pathlib import Path

class Config:
    def __init__(self):
        self.PROJECT_NAME = "EGHF-Net"
        self.VERSION_SUFFIX = "_DDD17_Target76_EIFNet_Align_316"

        self.PREPROCESSED_ROOT = Path(r"/home/lizhuoxian/DataSets/DDD17_Preprocessed_EGHF")
        # self.PREPROCESSED_ROOT = Path(r"F:\Downloads\datasets\DDD17-Events\dataset_DDD17-Events_our_codification\DDD17_Preprocessed_EGHF")
        self.PREPROCESSED_TRAIN = self.PREPROCESSED_ROOT / "train"
        self.PREPROCESSED_TEST = self.PREPROCESSED_ROOT / "test"

        self.PRETRAINED_RGB_PATH = Path("./pretrained/mit_b2.pth")
        self.PRETRAINED_EVT_PATH = Path("./pretrained/mit_b0.pth")

        self.RUNS_DIR = Path("./runs") / (self.PROJECT_NAME + self.VERSION_SUFFIX)
        self.CKPT_DIR = Path("./checkpoints") / self.VERSION_SUFFIX
        self.LOG_DIR = Path("./logs") / self.VERSION_SUFFIX
        self.TXT_LOG_DIR = Path("./txt_logs") / self.VERSION_SUFFIX
        self.LATEST_CKPT_PATH = self.CKPT_DIR / "latest_checkpoint.pth"
        self.BEST_MIOU_CKPT_PATH = self.CKPT_DIR / "best_miou_checkpoint.pth"

        for p in [self.RUNS_DIR, self.CKPT_DIR, self.LOG_DIR, self.TXT_LOG_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        self.RGB_BACKBONE = 'b2'
        self.EVENT_BACKBONE = 'b0'
        self.NUM_CLASSES = 6
        self.NUM_BINS = 3
        self.EVENT_INPUT_CHANS = self.NUM_BINS * 2
        self.IMG_SIZE = (200, 346)
        self.TRAIN_CROP_SIZE = (200, 346)

        self.DEVICE = "cuda:0"
        self.EPOCHS = 100
        self.BATCH_SIZE = 8
        self.ACCUMULATION_STEPS = 4
        self.LEARNING_RATE = 2e-4
        self.WEIGHT_DECAY = 0.01
        self.VAL_INTERVAL = 1

        self.ENABLE_COPY_PASTE = True
        self.OHEM_KEEP_RATIO = 1.0

config = Config()