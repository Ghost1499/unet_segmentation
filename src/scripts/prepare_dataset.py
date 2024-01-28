from concurrent.futures import ThreadPoolExecutor
import fnmatch
from pathlib import Path
from token import OP
from typing import Callable, Optional, Self

from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow, show
import numpy as np
import numpy.typing as npt

from skimage.transform import resize
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_opening,
)
from skimage.morphology.footprints import disk
from tqdm import tqdm

from src.ImagesDir import ImagesDir
from src.configs import ds_prepare_config, io_config

KERNEL_RADIUS = 3
OP_KERNEL = disk(KERNEL_RADIUS, dtype=np.bool_)  # type: ignore
CONT_MASKS_SAVE_EXT = ".jpg"


def create_contours_mask(mask: npt.NDArray[np.uint8]):
    mask = mask.astype(np.bool_)
    dilated = binary_dilation(binary_opening(mask, OP_KERNEL), OP_KERNEL)
    eroded = binary_erosion(binary_closing(mask, OP_KERNEL), OP_KERNEL)
    contours_mask = dilated & ~eroded
    return contours_mask.astype(np.uint8) * 255


def cont_mask_process(mask_load_path: Path, mask_save_path: Path) -> None:
    mask = imread(mask_load_path)
    mask = mask[..., 0]
    mask = np.where(mask >= 256 / 2, 255, 0)
    contours_mask = create_contours_mask(mask)
    imsave(mask_save_path.with_suffix(CONT_MASKS_SAVE_EXT), contours_mask)


def resize_img(img, is_mask):
    if is_mask:
        order = 0
        anti_aliasing = False
    else:
        order = 1
        anti_aliasing = True
    return resize(
        img, ds_prepare_config.TARGET_SIZE, order=order, anti_aliasing=anti_aliasing
    )


def resizing_process(img_load_path, img_save_path) -> None:
    img = imread(img_load_path)
    is_mask = "mask" in str(img_load_path.parent)
    resized = resize_img(img, is_mask)
    imsave(img_save_path, resized)


def process_images(
    load_folder: Path,
    save_folder: Path,
    process_fnc: Callable[[Path, Path], None],
    load_pattern: Optional[str] = None,
    rewrite: bool = True,
) -> None:
    load_folder = ImagesDir(load_folder)
    if not load_pattern:
        load_pattern = "*"
    load_paths_gen = load_folder.rglob(load_pattern)
    with ThreadPoolExecutor() as executor:
        for load_path in tqdm(load_paths_gen, desc="Image processing"):
            save_path = save_folder / load_path.relative_to(load_folder)
            if not save_path.parent.is_dir():
                save_path.parent.mkdir(parents=True)
            if save_path.exists() and not rewrite:
                continue
            executor.submit(process_fnc, load_path, save_path)


def test_resize():
    img_load_path = Path(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\test\0cdf5b5d0ce1_03.jpg"
    )
    img_save_path = Path(
        r"C:\CSF\programs\Segmentation_cont\results\0cdf5b5d0ce1_03.jpg"
    )
    resizing_process(img_load_path, img_save_path)


def test_cont_mask():
    image_path = Path(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana_mini\test_masks\0de66245f268_13_mask.gif"
    )
    mask = imread(image_path)
    mask = mask[..., 0]
    mask = np.where(mask >= 256 / 2, 255, 0)
    cont_mask = create_contours_mask(mask)
    imshow(cont_mask)
    show()
    imsave(Path("results") / (image_path.stem + ".png"), cont_mask)


def make_carvana_mini(rewrite):
    load_folder = io_config.CARVANA_DIR
    save_folder = io_config.CARVANA_MINI_DIR
    process_images(load_folder, save_folder, resizing_process, rewrite=rewrite)


def make_contours(rewrite):
    for split in ["train", "val", "test"]:
        load_folder = io_config.get_samples_dir(is_mini=True, split=split, mask="masks")
        save_folder = io_config.get_samples_dir(
            is_mini=True, split=split, mask="contours_insensitive"
        )
        process_images(load_folder, save_folder, cont_mask_process, rewrite=rewrite)


if __name__ == "__main__":
    # test_resize()
    # make_carvana_mini(True)
    make_contours(True)
    # test_cont_mask()
