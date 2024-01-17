from pathlib import Path

import numpy as np
from skimage.io import imread
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from configs import io_config, model_config, training_config, ds_prepare_config
from configs.training_config import COMPILE_CONFIGS

from utils.io_utils import paths_from_dir
from utils.io_utils import save_model


RNG = np.random.RandomState(ds_prepare_config.RANDOM_STATE)


def get_image_shapes(dir: Path):
    return imread(next(dir.iterdir())).shape


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
        paths_ds = tf.data.Dataset.from_tensor_slices(
            tuple(list(el) for el in zip(*paths))
        )

        def load_images_masks(image_path, mask_path):
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image, channels=3)  # type: ignore
            image = tf.image.resize(image, model_config.TARGET_SHAPE[0:2])
            image = image / 255.0  # type: ignore
            # image = tf.image.convert_image_dtype(image, dtype=tf.dtypes.float32)

            mask = tf.io.read_file(mask_path)
            mask = tf.io.decode_image(mask, channels=3, expand_animations=False)
            mask = tf.image.rgb_to_grayscale(mask)
            mask = tf.image.resize(
                mask, model_config.TARGET_SHAPE[0:2], method="nearest"
            )
            mask = tf.image.convert_image_dtype(mask, dtype=tf.dtypes.uint8)
            mask = mask / 255  # type: ignore

            return image, mask

        ds = (
            paths_ds.map(load_images_masks, num_parallel_calls=tf.data.AUTOTUNE)
            # .shuffle(
            #     ds_prepare_config.DS_SHUFFLE_BUFF_SIZE, ds_prepare_config.RANDOM_STATE
            # )
            .batch(ds_prepare_config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
        )

        ds_store[name] = ds

    optimaizer = keras.optimizers.Adam(
        learning_rate=training_config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False,
    )  # ,decay=1e-6)

    comp_config = COMPILE_CONFIGS[model_config.OUT_SIZE]
    comp_config["optimizer"] = optimaizer
    comp_config["run_eagerly"] = training_config.DEBUG_MODEL
    model.compile(**comp_config)

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
        shuffle=False,
    )
    model.save(io_config.MODEL_SAVE_DIR / f"{model_name}.keras")


if __name__ == "__main__":
    # pass
    # save_dataset_npy('carvana')
    save_model("unet0")
    train_model("unet0")
    # test_load_images_masks()
