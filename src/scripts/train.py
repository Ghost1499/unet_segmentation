from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard

from configs import io_config, model_config, training_config, ds_prepare_config
from configs.training.make_config import make_config
from configs.training_config import COMPILE_CONFIGS
from utils.io_utils import save_model_arch
from datastore import (
    DSPreparer,
    InMemoryDSPreparer,
    make_train_datastore,
    make_val_datastore,
)


class ModelTrainer(ABC):
    __training_modes = ["segmentation", "contours"]

    @property
    def architect_path(self):
        return io_config.MODEL_SAVE_DIR / f"{self._model_name}_architecture.json"

    @property
    def model_path(self):
        return io_config.MODEL_SAVE_DIR / f"{self._model_name}.keras"

    @property
    def training_mode(self):
        return self._training_mode

    @training_mode.setter
    def training_mode(self, value):
        if value not in self.__training_modes:
            raise ValueError(
                "Недопустимое значение режима обучения модели.",
                value,
                self.__training_modes,
            )
        self._training_mode = value

    def __init__(
        self,
        model_name: str,
        training_mode: str,
        ds_preparer: DSPreparer,
        training_config: dict,
    ) -> None:
        self._model_name = model_name
        self.training_mode = training_mode
        self.read_model(self._model_name, self.training_mode)
        self._ds_preparer = ds_preparer
        self._training_config = training_config

    def read_model(self, model_name: str, training_mode: str) -> None:
        self._model_name = model_name
        self.training_mode = training_mode
        with open(self.architect_path) as f:
            json_model = f.read()
        self._model: keras.models.Model = model_from_json(json_model)

    def train_model(self, load_ds=False, save=True) -> None:
        if not load_ds:
            self._ds_preparer.prepare()
            self._ds_preparer.save()
        else:
            self._ds_preparer.load()

        self._model.compile(**self._training_config.pop(["compile_params"]))

        X = self._ds_preparer.X
        y = self._ds_preparer.y
        
        if y is None:
            if X is None:
                raise Exception("Обучающий набор данных пустой")
            # add fit
        else:
            
        
        self._fit()

        if save:
            self.save_model()

    @abstractmethod
    def _fit(self):
        pass

    def save_model(self) -> None:
        self._model.save(self.model_path)


class NumpyModelTrainer(ModelTrainer):
    def __init__(
        self,
        model_name: str,
        training_mode: str,
        ds_preparer: InMemoryDSPreparer,
        training_config: dict,
    ) -> None:
        super().__init__(model_name, training_mode, ds_preparer, training_config)

    def _fit(self):
        X = self._ds_preparer.X
        y = self._ds_preparer.y

        checkpointer = ModelCheckpoint(
            filepath=str(
                io_config.CHECKPOINTS_SAVE_DIR / f"{self._model_name}_{{epoch}}.keras"
            ),
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        )
        tboard = TensorBoard(io_config.TENSORBOARD_LOG_DIR)  # type: ignore
        callbacks = [checkpointer, tboard]
        self._model.fit(
            X, y, shuffle=True, callbacks=callbacks, **self._training_config
        )


def train_model(model_name):
    with open(io_config.MODEL_SAVE_DIR / f"{model_name}_architecture.json") as f:
        json_model = f.read()
    model: keras.models.Model = model_from_json(json_model)
    # image_shape = get_images_shapes(io_config.TRAIN_IMAGES_DIR)

    train_ds = make_train_datastore(
        is_mini=True, mask_type="contours_insensitive"
    ).dataset
    val_ds = make_val_datastore(is_mini=True, mask_type="contours_insensitive").dataset

    model.compile(**comp_config)

    checkpointer = ModelCheckpoint(
        filepath=str(io_config.CHECKPOINTS_SAVE_DIR / f"{model_name}_{{epoch}}.keras"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    tboard = TensorBoard(io_config.TENSORBOARD_LOG_DIR)  # type: ignore
    model.fit(
        train_ds,
        epochs=training_config.EPOCHS,
        callbacks=[checkpointer, tboard],
        validation_data=val_ds,
        shuffle=False,
    )
    model.save(io_config.MODEL_SAVE_DIR / f"{model_name}.keras")


def main():
    trainer = NumpyModelTrainer(
        "unet0 contours insensitive",
        "contours",
        InMemoryDSPreparer(),
        make_config(False, "contours"),
    )
    trainer.train_model(True, True)


if __name__ == "__main__":
    main()
