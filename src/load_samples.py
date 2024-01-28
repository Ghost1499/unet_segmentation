from json import load
from typing import Any
import numpy as np
import tensorflow as tf

from configs import ds_prepare_config


def load_resize_image(path: str) -> Any:
    """
    Загружает цветное изображение .jpeg, расположенное в path. Уменьшает до размера TARGET_SIZE и нормирует к диапазону [0,1] float32.

    Args:
        path (str): путь к изображению.

    Returns:
        Any: изображение в виде тензора типа float32
    """
    image = load_image(path)
    image = tf.image.resize(image, ds_prepare_config.TARGET_SIZE)
    return image


def load_image(path: str) -> Any:
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)  # type: ignore
    return image


def load_resize_mask(path: str) -> Any:
    """
    Загружает цветное изображение, расположенное в path. Преобразует в серое, уменьшает до размера TARGET_SIZE и нормирует к диапазону [0,1] типа int8(?).

    Args:
        path (str): путь к изображению.

    Returns:
        Any: изображение в виде тензора типа int8(?)
    """
    mask = load_mask(path)
    mask = tf.image.resize(mask, ds_prepare_config.TARGET_SIZE, method="nearest")
    return mask


def load_mask(path: str) -> Any:
    mask = tf.io.read_file(path)
    mask = tf.io.decode_image(
        mask, channels=3, expand_animations=False, dtype=tf.dtypes.float32
    )
    mask = tf.expand_dims(tf.unstack(mask, axis=-1)[0], axis=-1)
    return mask


def __test():
    image = load_resize_image(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\train\0cdf5b5d0ce1_01.jpg"
    ).numpy()
    print(
        f"Image:{image.shape}, {image.dtype}. Values: {image.min()}-{image.max()} [{image.mean()}]"
    )
    mask = load_resize_mask(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\train_masks\0cdf5b5d0ce1_01_mask.gif"
    ).numpy()
    print(f"Mask:{mask.shape}, {mask.dtype}. Values: {np.unique(mask)} [{mask.mean()}]")

    image = load_resize_image(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana_mini\train\0cdf5b5d0ce1_01.jpg"
    ).numpy()
    print(
        f"Image:{image.shape}, {image.dtype}. Values: {image.min()}-{image.max()} [{image.mean()}]"
    )
    mask = load_resize_mask(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana_mini\train_masks\0cdf5b5d0ce1_01_mask.gif"
    ).numpy()
    print(f"Mask:{mask.shape}, {mask.dtype}. Values: {np.unique(mask)} [{mask.mean()}]")


if __name__ == "__main__":
    __test()
