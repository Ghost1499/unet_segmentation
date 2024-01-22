from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import tensorflow as tf
import numpy.typing as npt

from utils.io_utils import get_sample_paths
from configs import ds_prepare_config


class Datastore(ABC):
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

    def __init__(self, images_dir: Path, masks_dir: Path, save_dir: Path) -> None:
        self._images_dir = images_dir
        self._masks_dir = masks_dir
        self._save_dir = save_dir
        # self.dataset = tf.data.Dataset.from

    # @abstractmethod
    # def load(self):
    #     pass

    # @abstractmethod
    # def save(self):
    #     pass


class PreshuffleDatastore(Datastore):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        save_dir: Path,
        load_image_fnc: Callable[[str], npt.NDArray],
        load_mask_fnc: Callable[[str], npt.NDArray],
        preloading_shuffle: bool = True,
        shuffle: bool = True,
        random_state=None,
    ) -> None:
        super().__init__(images_dir, masks_dir, save_dir)
        self.__load_image_fnc = load_image_fnc
        self.__load_mask_fnc = load_mask_fnc
        self._preloading_shuffle = preloading_shuffle
        self._shuffle = shuffle
        self.__random_state = random_state

        self.dataset = self.__load_dataset()

    def __load_dataset(self):
        image_paths, mask_paths = get_sample_paths(
            self._images_dir,
            self._masks_dir,
            self._preloading_shuffle,
            self.__random_state,
        )
        ds = tf.data.Dataset.zip(
            self.__load_images(image_paths), self.__load_masks(mask_paths)
        )
        if self._shuffle:
            ds = ds.shuffle(ds_prepare_config.DS_SHUFFLE_BUFF_SIZE, self.__random_state)

        ds = ds.batch(
            ds_prepare_config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return ds

    def __load_images(self, paths_ds):
        return paths_ds.map(self.__load_image_fnc, num_parallel_calls=tf.data.AUTOTUNE)

    def __load_masks(self, paths_ds):
        return paths_ds.map(self.__load_mask_fnc, num_parallel_calls=tf.data.AUTOTUNE)
