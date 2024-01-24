from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Generator, Union

from keras.utils import plot_model
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split

from configs import ds_prepare_config, io_config, model_config
from models import createUNetModel_My

from tqdm import tqdm


def paths_from_dir(
    dir: Path, extenstions=None, is_filename=False
) -> Generator[Union[str, Path], Any, None]:
    if not dir.is_dir():
        raise ValueError("Путь не является папкой")
    for img_path in dir.iterdir():
        if extenstions is not None:
            if img_path.suffix not in extenstions:
                continue
        yield img_path.name if is_filename else img_path


def move_samples(take_rate, images_dir, masks_dir) -> None:
    """
    Создает тестовую выборку в соотвествии с глобальной переменной размера тестовой выборки. Перемещает изображения и маски из тренировочных папок в тестовые.
    """
    image_paths = list(paths_from_dir(io_config.TRAIN_IMAGES_DIR))
    mask_paths = list(paths_from_dir(io_config.TRAIN_MASKS_DIR))

    ds = list(zip(image_paths, mask_paths, strict=True))
    taken_count = int(take_rate * len(ds))
    ds_prepare_config.RNG.shuffle(ds)  # type: ignore
    taken_ds = ds[:taken_count]

    def move_taken_element(image_path: Path, mask_path: Path):
        image_path.rename(images_dir / image_path.name)
        mask_path.rename(masks_dir / mask_path.name)

    [
        move_taken_element(image_path, mask_path)  # type: ignore
        for image_path, mask_path in tqdm(
            taken_ds, desc="Перемещение отделённых образцов", total=taken_count
        )
    ]


def save_dataset_npy(dataset_name):
    # print(*(list(filenames_from_dir(IMAGES_DIR))[:10]),sep='\n')
    image_paths = list(paths_from_dir(io_config.TRAIN_IMAGES_DIR))
    mask_paths = list(paths_from_dir(io_config.TRAIN_MASKS_DIR))

    assert len(image_paths) == len(mask_paths)
    images_number = len(image_paths)
    # image_shape = images[0].shape

    images = [
        imread(filename)
        for filename in tqdm(image_paths, desc="images reading", total=images_number)
    ]
    masks = [
        imread(filename)
        for filename in tqdm(mask_paths, desc="masks reading", total=images_number)
    ]
    # images_coll = imread_collection(image_paths)
    # masks_coll = imread_collection(mask_paths)

    idx = np.arange(images_number)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=ds_prepare_config.TEST_SIZE,
        random_state=ds_prepare_config.RNG,
        shuffle=True,
    )
    train_images = images[train_idx]
    test_images = images[test_idx]
    train_masks = masks[train_idx]
    test_masks = masks[test_idx]

    np.savez(
        io_config.CARVANA_DIR / f"{dataset_name}.npz",
        X_train=np.array(train_images),
        X_test=np.array(test_images),
        y_train=np.array(train_masks),
        y_test=np.array(test_masks),
    )


def save_model(model_name):
    # image_shape = get_images_shapes(io_config.TRAIN_IMAGES_DIR)
    model = createUNetModel_My(
        model_config.TARGET_SHAPE,
        model_config.NUMBER_CONVS,
        model_config.CONV_FILTERS,
        model_config.OUT_SIZE,
        model_config.L2_VALUE,
        model_config.DROPOUT_VALUE,
        model_config.BATCH_NORM,
    )

    with open(io_config.MODEL_SAVE_DIR / f"{model_name}.txt", "w") as f:
        with redirect_stdout(f):
            model.summary(show_trainable=True)

    model_json = model.to_json()
    with open(
        io_config.MODEL_SAVE_DIR / f"{model_name}_architecture.json", "w"
    ) as json_file:
        json_file.write(model_json)

    plot_model(
        model,
        str(io_config.MODEL_SAVE_DIR / f"{model_name}.png"),
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
    )


def get_sample_paths(
    images_folder: Path, masks_folder: Path, shuffle: bool, random_state
):
    image_paths = [str(path) for path in sorted(paths_from_dir(images_folder))]
    mask_paths = [str(path) for path in sorted(paths_from_dir(masks_folder))]

    if shuffle:
        paths = list(zip(image_paths, mask_paths, strict=True))
        rng = np.random.default_rng(random_state)
        rng.shuffle(paths)  # type: ignore
        image_paths, mask_paths = tuple(list(el) for el in zip(*paths))
    return image_paths, mask_paths


def get_image_shapes(dir: Path):
    return imread(next(dir.iterdir())).shape
