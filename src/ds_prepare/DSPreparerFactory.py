from configs import io_config, ds_prepare_config
from ds_prepare.datastore import DSPreparer, InMemoryDSPreparer


from abc import ABC, abstractmethod
from pathlib import Path


class DSPreparerFactory(ABC):
    def __init__(self, images_dir, masks_dir, ds_path) -> None:
        self._images_dir = images_dir
        self._masks_dir = masks_dir
        self._ds_path = ds_path

    @abstractmethod
    def _create(self) -> DSPreparer:
        pass

    def create_ds_preparer(self):
        return self._create()


class InMemPrepFactory(DSPreparerFactory):
    def __init__(self, images_dir, masks_dir, ds_path, random_state, shuffle) -> None:
        super().__init__(images_dir, masks_dir, ds_path)
        self._random_state = random_state
        self._shuffle = shuffle

    def _create(self) -> DSPreparer:
        return InMemoryDSPreparer(
            self._images_dir,
            self._masks_dir,
            self._ds_path,
            self._random_state,
            self._shuffle,
        )
