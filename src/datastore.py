from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable
from matplotlib.pyplot import imshow, show, figure

import tensorflow as tf
import numpy.typing as npt
from load_samples import load_image, load_mask

from utils.io_utils import get_sample_paths
from configs import ds_prepare_config, io_config


class Datastore(ABC):
    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, value) -> None:
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
            self.__load_elements(image_paths, self.__load_image_fnc),
            self.__load_elements(mask_paths, self.__load_mask_fnc),
        ).cache()
        if self._shuffle:
            ds = ds.shuffle(
                ds_prepare_config.DS_SHUFFLE_BUFF_SIZE,
                self.__random_state,
                reshuffle_each_iteration=True,
            )

        ds = ds.batch(
            ds_prepare_config.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return ds

    def __load_elements(self, paths, load_fnc):
        return tf.data.Dataset.from_tensor_slices(paths).map(
            load_fnc, num_parallel_calls=tf.data.AUTOTUNE
        )


def make_train_datastore() -> PreshuffleDatastore:
    return PreshuffleDatastore(
        io_config.TRAIN_IMAGES_DIR,
        io_config.TRAIN_MASKS_DIR,
        Path(),
        load_image,
        load_mask,
        random_state=ds_prepare_config.RANDOM_STATE,
    )


def make_val_datastore() -> PreshuffleDatastore:
    return PreshuffleDatastore(
        io_config.VAL_IMAGES_DIR,
        io_config.VAL_MASKS_DIR,
        Path(),
        load_image,
        load_mask,
        random_state=ds_prepare_config.RANDOM_STATE,
        shuffle=False,
    )


def make_test_datastore():
    return PreshuffleDatastore(
        io_config.TEST_IMAGES_DIR,
        io_config.TEST_MASKS_DIR,
        Path(),
        load_image,
        load_mask,
        random_state=ds_prepare_config.RANDOM_STATE,
        shuffle=False,
    )


def __test():
    datastore = make_test_datastore()
    ds = datastore.dataset
    image, mask = next(ds.unbatch().take(1).as_numpy_iterator())
    figure()
    imshow(image)
    show()
    figure()
    imshow(mask)
    show()


if __name__ == "__main__":
    __test()
