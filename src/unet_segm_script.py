from pathlib import Path
from contextlib import redirect_stdout
from typing import Any, Generator, Union

import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
import keras
from keras.utils import plot_model
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from tqdm import tqdm
from configs import io_config, model_config, training_config, ds_prepare_config

from models import createUNetModel_My


RNG = np.random.RandomState(ds_prepare_config.RANDOM_STATE)


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


def take_test_val_samples(take_rate, images_dir, masks_dir) -> None:
    """
    Создает тестовую выборку в соотвествии с глобальной переменной размера тестовой выборки. Перемещает изображения и маски из тренировочных папок в тестовые.
    """
    image_paths = list(paths_from_dir(io_config.TRAIN_IMAGES_DIR))
    mask_paths = list(paths_from_dir(io_config.TRAIN_MASKS_DIR))

    ds = list(zip(image_paths, mask_paths, strict=True))
    taken_count = int(take_rate * len(ds))
    RNG.shuffle(ds)  # type: ignore
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

    # ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    # ds = ds.shuffle(ds.cardinality(), RANDOM_STATE)

    # test_ds = ds.take(test_number)


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
        idx, test_size=ds_prepare_config.TEST_SIZE, random_state=RNG, shuffle=True
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


def get_images_shapes(dir: Path):
    return imread(next(dir.iterdir())).shape


def save_model(model_name):
    # image_shape = get_images_shapes(io_config.TRAIN_IMAGES_DIR)
    model: keras.Model = createUNetModel_My(
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


def train_model(model_name):
    with open(io_config.MODEL_SAVE_DIR / f"{model_name}_architecture.json") as f:
        json_model = f.read()
    model: keras.models.Model = model_from_json(json_model)
    # image_shape = get_images_shapes(io_config.TRAIN_IMAGES_DIR)

    ds_store = {}
    for name, (images_dir, masks_dir) in {
        "train": (io_config.TRAIN_IMAGES_DIR, io_config.TRAIN_MASKS_DIR),
        "val": (io_config.VAL_IMAGES_DIR, io_config.VAL_MASKS_DIR),
    }.items():
        image_paths = [str(path) for path in paths_from_dir(images_dir)]
        mask_paths = [str(path) for path in paths_from_dir(masks_dir)]

        paths = list(zip(image_paths, mask_paths, strict=True))
        RNG.shuffle(paths)  # type: ignore
        paths_ds = tf.data.Dataset.from_tensor_slices(tuple(list(el) for el in zip(*paths)))

        def load_images_masks(image_path, mask_path):
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image,channels=3)
            image = tf.image.resize(image, model_config.TARGET_SHAPE[0:2])
            image = tf.image.convert_image_dtype(image, dtype=tf.dtypes.float32)

            mask = tf.io.read_file(mask_path)
            mask = tf.io.decode_image(mask, channels=3, expand_animations=False)
            mask = tf.image.rgb_to_grayscale(mask)
            mask = tf.image.resize(mask, model_config.TARGET_SHAPE[0:2])
            mask = tf.image.convert_image_dtype(mask, dtype=tf.dtypes.uint8)

            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            return image, mask

        ds = (
            paths_ds.map(load_images_masks, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(
                ds_prepare_config.DS_SHUFFLE_BUFF_SIZE, ds_prepare_config.RANDOM_STATE
            )
            .batch(ds_prepare_config.BATCH_SIZE)
        )

        ds_store[name] = ds

    opt = keras.optimizers.Adam(
        learning_rate=training_config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False,
    )  # ,decay=1e-6)

    if model_config.OUT_SIZE == 1:
        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
    elif model_config.OUT_SIZE == 2:
        model.compile(
            optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
    else:
        model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
        )

    checkpointer = ModelCheckpoint(
        filepath=io_config.CHECKPOINTS_SAVE_DIR / f"{model_name}_{{epoch}}.keras",
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    tboard = TensorBoard(io_config.TENSORBOARD_LOG_DIR)  # type: ignore
    model.fit(
        ds_store["train"],
        epochs=training_config.EPOCHS,
        callbacks=[checkpointer, tboard],
        validation_data=ds_store["val"],
        shuffle=False
    )
    model.save(io_config.MODEL_SAVE_DIR / f"{model_name}.keras")


if __name__ == "__main__":
    # pass
    # save_dataset_npy('carvana')
    # save_model("unet0")
    train_model("unet0")
