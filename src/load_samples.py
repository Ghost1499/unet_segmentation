from json import load
from typing import Any
import numpy as np
import tensorflow as tf

from configs import model_config


def load_image(path: str) -> Any:
    """
    Загружает цветное изображение .jpeg, расположенное в path. Уменьшает до размера TARGET_SIZE и нормирует к диапазону [0,1] float32.

    Args:
        path (str): путь к изображению.

    Returns:
        Any: изображение в виде тензора типа float32
    """
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)  # type: ignore
    image = tf.image.resize(image, model_config.TARGET_SHAPE[0:2])
    image = image / 255.0  # type: ignore
    return image


def load_mask(path: str) -> Any:
    """
    Загружает цветное изображение, расположенное в path. Преобразует в серое, уменьшает до размера TARGET_SIZE и нормирует к диапазону [0,1] типа int8(?).

    Args:
        path (str): путь к изображению.

    Returns:
        Any: изображение в виде тензора типа int8(?)
    """
    mask = tf.io.read_file(path)
    mask = tf.io.decode_image(mask, channels=3, expand_animations=False)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, model_config.TARGET_SHAPE[0:2], method="nearest")
    mask = tf.image.convert_image_dtype(mask, dtype=tf.dtypes.uint8)
    mask = mask / 255  # type: ignore
    return mask


def __test():
    image = load_image(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\train\0cdf5b5d0ce1_01.jpg"
    ).numpy()
    print(
        f"Image:{image.shape}, {image.dtype}. Values: {image.min()}-{image.max()} [{image.mean()}]"
    )
    mask = load_mask(
        r"C:\CSF\programs\Segmentation_cont\data\datasets\carvana\train_masks\0cdf5b5d0ce1_01_mask.gif"
    ).numpy()
    print(f"Mask:{mask.shape}, {mask.dtype}. Values: {np.unique(mask)} [{mask.mean()}]")


if __name__ == "__main__":
    __test()
