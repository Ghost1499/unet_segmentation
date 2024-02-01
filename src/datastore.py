from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable

from matplotlib.pyplot import imread, imshow, show, figure
import numpy as np
import tensorflow as tf
import numpy.typing as npt
from skimage.util import img_as_float32

from ImagesDir import ImagesDir
from load_samples import load_image, load_resize_image, load_mask, load_resize_mask
from utils.io_utils import get_sample_paths, shuffle_paths
from configs import ds_prepare_config, io_config


def shuffle_sample_paths(
    image_paths: Iterable[str | Path], mask_paths: Iterable[str | Path], random_state
) -> tuple[list[str | Path], list[str | Path]]:
    image_paths_sh, mask_paths_sh = shuffle_paths(
        image_paths, mask_paths, random_state=random_state
    )
    return image_paths_sh, mask_paths_sh


class DSPreparer(ABC):
    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value) -> None:
        self._dataset = value

    def __init__(self, images_dir: Path, masks_dir: Path) -> None:
        self._images_dir = ImagesDir(images_dir)
        self._masks_dir = ImagesDir(masks_dir)
        # self._save_dir = save_dir
        # self.dataset = tf.data.Dataset.from

    @abstractmethod
    def prepare(self):
        pass

    # @abstractmethod
    # def load(self):
    #     pass

    @abstractmethod
    def save(self) -> None:
        pass


class InMemoryDSPreparer(DSPreparer):
    __img_load_fun: Callable[[str | Path], npt.NDArray] = imread

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    def __init__(
        self, images_dir: Path, masks_dir: Path, random_state: int, shuffle=True
    ) -> None:
        super().__init__(images_dir, masks_dir)
        self._X = np.array([])
        self._y = np.array([])
        self._random_state = random_state
        self._shuffle = shuffle

    def prepare(self):
        image_paths = sorted(self._images_dir.iterdir())
        mask_paths = sorted(self._masks_dir.iterdir())

        if self._shuffle:
            image_paths, mask_paths = shuffle_sample_paths(
                image_paths, mask_paths, self._random_state
            )

        images = [self._load_image(p) for p in image_paths]
        masks = [self._load_mask(p) for p in mask_paths]

        self._X = np.array(images)
        self._y = np.array(masks)

    @classmethod
    def _load_image(cls, path) -> npt.NDArray[Any]:
        return img_as_float32(cls.__img_load_fun(path))

    @classmethod
    def _load_mask(cls, path):
        return img_as_float32(cls.__img_load_fun(path)[..., 0])


class PreshuffleDatastore(DSPreparer):
    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        load_image_fnc: Callable[[str], npt.NDArray],
        load_mask_fnc: Callable[[str], npt.NDArray],
        preloading_shuffle: bool = True,
        shuffle: bool = True,
        random_state=None,
    ) -> None:
        super().__init__(images_dir, masks_dir)
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


def make_datastore(split, is_mini, mask_type) -> PreshuffleDatastore:
    images_dir = io_config.get_samples_dir(is_mini, split=split, mask=None)
    masks_dir = io_config.get_samples_dir(is_mini, split=split, mask=mask_type)
    load_img_fnc, load_mask_fnc = (
        (load_image, load_mask) if is_mini else (load_resize_image, load_resize_mask)
    )
    shuffle = split == "train"
    return PreshuffleDatastore(
        images_dir,
        masks_dir,
        load_img_fnc,
        load_mask_fnc,
        random_state=ds_prepare_config.RANDOM_STATE,
        shuffle=shuffle,
    )


def make_train_datastore(is_mini, mask_type: str) -> PreshuffleDatastore:
    return make_datastore("train", is_mini, mask_type)


def make_val_datastore(is_mini, mask_type: str) -> PreshuffleDatastore:
    return make_datastore("val", is_mini, mask_type)


def make_test_datastore(is_mini, mask_type: str) -> PreshuffleDatastore:
    return make_datastore("test", is_mini, mask_type)


def __test():
    datastore = make_train_datastore(is_mini=True, mask_type="contours_insensitive")
    ds = datastore.dataset
    image, mask = next(ds.unbatch().take(1).as_numpy_iterator())  # type: ignore
    figure()
    imshow(image)
    show()
    figure()
    imshow(mask)
    show()


if __name__ == "__main__":
    __test()
