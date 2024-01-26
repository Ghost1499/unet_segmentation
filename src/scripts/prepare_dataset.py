import fnmatch
from pathlib import Path
from typing import Self

from matplotlib.image import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

from src.ImagesDir import ImagesDir
from src.configs import ds_prepare_config, io_config


def resizing_process(img_load_path, img_save_path):
    img = imread(img_load_path)
    resized = resize(img, ds_prepare_config.TARGET_SIZE)
    imsave(img_save_path, resized)


def make_resized(load_folder: Path, save_folder: Path):
    load_folder = ImagesDir(load_folder)
    load_paths_gen = load_folder.rglob("*")
    for load_path in tqdm(load_paths_gen, desc="Making resized"):
        save_path = save_folder / load_path.relative_to(load_folder)
        if not save_path.parent.is_dir():
            save_path.parent.mkdir(parents=True)
        resizing_process(load_path, save_path)


def test():
    img_load_path = Path(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\test\0cdf5b5d0ce1_03.jpg"
    )
    img_save_path = Path(
        r"C:\CSF\programs\Segmentation_cont\results\0cdf5b5d0ce1_03.jpg"
    )
    resizing_process(img_load_path, img_save_path)


def main():
    load_folder = io_config.CARVANA_DIR
    save_folder = load_folder.with_name("_".join([load_folder.name, "mini"]))
    make_resized(load_folder, save_folder)


if __name__ == "__main__":
    main()
    # test()
