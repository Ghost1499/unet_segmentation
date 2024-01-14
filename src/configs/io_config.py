from pathlib import Path


DATASETS_DIR = Path(r"data/datasets")
CARVANA_DIR = DATASETS_DIR / "carvana"
TRAIN_IMAGES_DIR = CARVANA_DIR / "train"
TRAIN_MASKS_DIR = CARVANA_DIR / "train_masks"
VAL_IMAGES_DIR = CARVANA_DIR / "val"
VAL_MASKS_DIR = CARVANA_DIR / "val_masks"
TEST_IMAGES_DIR = CARVANA_DIR / "test"
TEST_MASKS_DIR = CARVANA_DIR / "test_masks"
MODEL_SAVE_DIR = Path(r"data/models")
CHECKPOINTS_SAVE_DIR = MODEL_SAVE_DIR / "checkpoints"
TENSORBOARD_LOG_DIR = Path(r"data/logs")
