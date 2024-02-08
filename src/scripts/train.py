from pathlib import Path
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard

import sys

sys.path.insert(0, str(Path("src").absolute()))

from configs import io_config, ds_prepare_config
from configs.make_train_config import make_config
from datastore import (
    DSPreparer,
    InMemoryDSPreparer,
)


class ModelTrainer:
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
            self._ds_preparer.save(True)
        else:
            self._ds_preparer.load()

        self._model.compile(**self._training_config.pop("compile_params"))

        X = self._ds_preparer.X
        y = self._ds_preparer.y

        if y is None and X is None:
            raise Exception("Обучающий набор данных пустой")

        fit_kwargs = self._make_fit_kwargs(X, y)

        self._model.fit(**fit_kwargs)

        if save:
            self.save_model()

    def save_model(self) -> None:
        self._model.save(self.model_path)

    def _create_checkpointer(self, **kwargs):
        return ModelCheckpoint(
            io_config.CHECKPOINTS_SAVE_DIR / f"{self._model_name}_{{epoch}}.keras",
            **kwargs,
        )

    def _create_tensorboard(self, **kwargs):
        return TensorBoard(str(io_config.TENSORBOARD_LOG_DIR), **kwargs)

    def _make_fit_kwargs(self, X, y):
        callbacks = [
            self._create_checkpointer(
                **self._training_config.pop("checkpointer_params")
            ),
            self._create_tensorboard(),
        ]
        fit_kwargs = self._training_config.pop("fit_params")
        fit_kwargs["x"] = X
        fit_kwargs["y"] = y
        fit_kwargs["callbacks"] = callbacks
        fit_kwargs["shuffle"] = True
        return fit_kwargs


def test():
    mode = "contours"
    trainer = ModelTrainer(
        "unet0 contours insensitive",
        mode,
        InMemoryDSPreparer(
            io_config.get_samples_dir(True, "test"),
            io_config.get_samples_dir(True, "test", "contours_insensitive"),
            "results",
            ds_prepare_config.RANDOM_STATE,
            True,
        ),
        make_config(False, mode),
    )
    trainer.train_model(True, True)


if __name__ == "__main__":
    test()
