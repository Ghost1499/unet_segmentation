import tensorflow as tf
from pathlib import Path
from configs import ds_prepare_config, io_config, model_config
from utils.io_utils import get_sample_paths


def load_image_mask(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)  # type: ignore
    image = tf.image.resize(image, model_config.TARGET_SHAPE[0:2])
    image = image / 255.0  # type: ignore
    # image = tf.image.convert_image_dtype(image, dtype=tf.dtypes.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_image(mask, channels=3, expand_animations=False)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, model_config.TARGET_SHAPE[0:2], method="nearest")
    mask = tf.image.convert_image_dtype(mask, dtype=tf.dtypes.uint8)
    mask = mask / 255  # type: ignore

    return image, mask


def get_dataset(
    images_foler: Path, masks_folder: Path, prepare_shuffle=True, training_shuffle=True
):
    image_paths, mask_paths = get_sample_paths(
        images_foler, masks_folder, prepare_shuffle
    )
    paths_ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = paths_ds.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    if training_shuffle:
        ds = ds.shuffle(
            ds_prepare_config.DS_SHUFFLE_BUFF_SIZE, ds_prepare_config.RANDOM_STATE
        )
    ds = ds.batch(
        ds_prepare_config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    return ds


def get_test_ds():
    return get_dataset(
        io_config.TEST_IMAGES_DIR, io_config.TEST_MASKS_DIR, training_shuffle=False
    )


def get_val_ds():
    return get_dataset(
        io_config.VAL_IMAGES_DIR, io_config.VAL_MASKS_DIR, training_shuffle=False
    )


def get_train_ds():
    return get_dataset(io_config.TRAIN_IMAGES_DIR, io_config.TRAIN_MASKS_DIR)
