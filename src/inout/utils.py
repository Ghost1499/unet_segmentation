from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Generator, Iterable, Union

from tqdm import tqdm

from keras.utils import plot_model
import numpy as np
from skimage.io import imread

from models import createUNetModel_My
from configs import model_config, io_config, ds_prepare_config


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


def save_model_arch(model_name):
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


def shuffle_paths(
    *paths_secs: Iterable[Path | str | Any], random_state
) -> tuple[list[Any], ...]:
    try:
        zp = zip(*paths_secs, strict=True)
    except ValueError as err:
        raise ValueError(err, "Итерируемые объекты имеют неодинаковую длину")
    paths = list(zp)
    rng = np.random.default_rng(random_state)
    rng.shuffle(paths)  # type: ignore
    shuffled = tuple(list(el) for el in zip(*paths))
    return shuffled


def get_image_shapes(dir: Path):
    return imread(next(dir.iterdir())).shape


def test():
    print(*shuffle_paths([1, 2, 3], [4, 5, 6], random_state=0))
    print(*shuffle_paths([], [], random_state=0))
    print(*shuffle_paths([], random_state=0))


if __name__ == "__main__":
    test()
