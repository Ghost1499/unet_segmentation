import fnmatch
from pathlib import Path
from typing import Callable, Self

from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow, show
import numpy as np
import numpy.typing as npt

from skimage.transform import resize
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology.footprints import disk
from tqdm import tqdm

from src.ImagesDir import ImagesDir
from src.configs import ds_prepare_config, io_config

KERNEL_RADIUS = 3
OP_KERNEL = disk(KERNEL_RADIUS, dtype=np.bool_)
CONT_MASKS_SAVE_EXT = ".jpg"


def create_contours_mask(mask: npt.NDArray[np.uint8]):
    mask = mask.astype(np.bool_)
    dilated = binary_dilation(mask, OP_KERNEL)
    eroded = binary_erosion(mask, OP_KERNEL)
    contours_mask = dilated & ~eroded
    return contours_mask.astype(np.uint8) * 255


def cont_mask_process(mask_load_path: Path, mask_save_path: Path) -> None:
    mask = imread(mask_load_path)
    mask = mask[..., 0]
    mask = np.where(mask >= 256 / 2, 255, 0)
    contours_mask = create_contours_mask(mask)
    imsave(mask_save_path.with_suffix(CONT_MASKS_SAVE_EXT), contours_mask)


def resizing_process(img_load_path, img_save_path) -> None:
    img = imread(img_load_path)
    resized = resize(img, ds_prepare_config.TARGET_SIZE)
    imsave(img_save_path, resized)


def process_images(
    load_folder: Path, save_folder: Path, process_fnc: Callable[[Path, Path], None]
) -> None:
    load_folder = ImagesDir(load_folder)
    load_paths_gen = load_folder.rglob("*")
    for load_path in tqdm(load_paths_gen, desc="Making resized"):
        save_path = save_folder / load_path.relative_to(load_folder)
        if not save_path.parent.is_dir():
            save_path.parent.mkdir(parents=True)
        process_fnc(load_path, save_path)


def test():
    img_load_path = Path(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\test\0cdf5b5d0ce1_03.jpg"
    )
    img_save_path = Path(
        r"C:\CSF\programs\Segmentation_cont\results\0cdf5b5d0ce1_03.jpg"
    )
    resizing_process(img_load_path, img_save_path)


def test_cont_mask():
    mask = imread(
        Path(
            r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana_mini\test_masks\0cdf5b5d0ce1_03_mask.gif"
        )
    )
    mask = mask[..., 0]
    mask = np.where(mask >= 256 / 2, 255, 0)
    cont_mask = create_contours_mask(mask)
    imshow(cont_mask)
    show()
    imsave("results/0cdf5b5d0ce1_03_mask_mini.png", cont_mask)


def main():
    load_folder = io_config.CARVANA_DIR
    save_folder = load_folder.with_name("_".join([load_folder.name, "mini"]))
    process_images(load_folder, save_folder)


if __name__ == "__main__":
    # main()
    test_cont_mask()
    # test()
