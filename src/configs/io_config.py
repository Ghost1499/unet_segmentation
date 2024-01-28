from pathlib import Path


DATASETS_DIR = Path(r"data/datasets")
CARVANA_DIR = DATASETS_DIR / "carvana"
CARVANA_MINI_DIR = DATASETS_DIR / "carvana_mini"
TRAIN_IMAGES_DIR = CARVANA_DIR / "train"
TRAIN_MASKS_DIR = CARVANA_DIR / "train_masks"
VAL_IMAGES_DIR = CARVANA_DIR / "val"
VAL_MASKS_DIR = CARVANA_DIR / "val_masks"
TEST_IMAGES_DIR = CARVANA_DIR / "test"
TEST_MASKS_DIR = CARVANA_DIR / "test_masks"
MODEL_SAVE_DIR = Path(r"data/models")
CHECKPOINTS_SAVE_DIR = MODEL_SAVE_DIR / "checkpoints"
TENSORBOARD_LOG_DIR = Path(r"data/logs")

__splits = ("all", "train", "val", "test")
__mask_types = (None, "masks", "contours", "contours_insensitive")
__name_sep = "_"


def get_samples_dir(is_mini=False, split: str = "all", mask=None):
    if split not in __splits:
        raise ValueError(split)
    if mask not in __mask_types:
        raise ValueError(mask)
    res_dir = CARVANA_MINI_DIR if is_mini else CARVANA_DIR
    if split != "all":
        if mask is not None:
            folder = __name_sep.join([split, mask])
        else:
            folder = split
        res_dir = res_dir / folder
    return res_dir


def test():
    print(
        get_samples_dir(),
        get_samples_dir(is_mini=True),
        get_samples_dir(split="train"),
        get_samples_dir(split="val", mask="mask"),
        get_samples_dir(is_mini=True, split="test", mask="contour"),
        sep="\n",
    )


if __name__ == "__main__":
    test()
