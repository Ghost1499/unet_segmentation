from pathlib import Path
from typing import Dict
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard

if __name__ == "__main__":
    import sys

    sys.path.append(str(Path("src").absolute()))

from model.ModelManager import ModelManager
from ds_prepare.ds_prep_fact import create_train_fact, create_val_fact
from configs import io_config
from configs.make_train_config import make_config


class ModelTrainer:
    __training_modes = ["segmentation", "contours", "contours_ls"]

    @property
    def model_name(self):
        return self._model_manager.model_name

    @model_name.setter
    def model_name(self, value):
        self._model_manager.model_name = value

    @property
    def training_mode(self):
        return self._model_manager.model_mode

    @training_mode.setter
    def training_mode(self, value) -> None:
        if value not in self.__training_modes:
            raise ValueError(
                "Недопустимое значение режима обучения модели.",
                value,
                self.__training_modes,
            )
        self._model_manager.model_mode = value

    def __init__(self, model_name: str, training_mode: str, is_debug=False) -> None:
        self._model_manager = ModelManager(model_name, training_mode)
        self.read_model()
        self._train_ds_preparer = create_train_fact(
            self.training_mode
        ).create_ds_preparer()
        self._val_ds_preparer = create_val_fact(self.training_mode).create_ds_preparer()
        self._training_config = make_config(is_debug, training_mode)

    def read_model(self) -> None:
        self._model: keras.models.Model = self._model_manager.read_model()

    def train_model(self, load_ds=False, save_ds=True) -> None:
        self._model.compile(**self._training_config.pop("compile_params"))

        self._prepare_data(load_ds, save_ds)

        fit_kwargs = self._make_fit_kwargs()
        self._model.fit(**fit_kwargs)

        self._save_model()

    def _save_model(self, overwrite=True) -> None:
        self._model_manager.save_model(self._model, overwrite)

    def _create_checkpointer(self, **kwargs) -> ModelCheckpoint:
        return ModelCheckpoint(
            io_config.CHECKPOINTS_SAVE_DIR / f"{self.model_name}_{{epoch}}.keras",
            **kwargs,
        )

    def _create_tensorboard(self, **kwargs) -> TensorBoard:
        return TensorBoard(str(io_config.TENSORBOARD_LOG_DIR), **kwargs)

    def _prepare_data(self, load_ds, save_ds) -> None:
        ds_preps = [self._train_ds_preparer]
        if self._val_ds_preparer:
            ds_preps.append(self._val_ds_preparer)
        for ds_preparer in ds_preps:
            if not load_ds:
                ds_preparer.prepare()
                if save_ds:
                    ds_preparer.save(True)
            else:
                ds_preparer.load()

        if self._train_ds_preparer.X is None and self._train_ds_preparer.y is None:
            raise Exception("Обучающий набор данных пустой")

    def _make_fit_kwargs(self) -> Dict:
        fit_kwargs = self._training_config.pop("fit_params")

        fit_kwargs["x"] = self._train_ds_preparer.X
        if self._train_ds_preparer.y is not None:
            fit_kwargs["y"] = self._train_ds_preparer.y

        # иначе используется validation_split
        if self._val_ds_preparer is not None:
            val_data = (
                (
                    self._val_ds_preparer.X,
                    self._val_ds_preparer.y,
                )
                if self._val_ds_preparer.y is not None
                else (self._val_ds_preparer.X)
            )
            fit_kwargs["validation_data"] = val_data

        callbacks = [
            self._create_checkpointer(
                **self._training_config.pop("checkpointer_params")
            ),
            self._create_tensorboard(),
        ]
        fit_kwargs["callbacks"] = callbacks
        fit_kwargs["shuffle"] = True
        return fit_kwargs


def test() -> None:
    model = "unet0bn"
    mode = "contours_ls"
    trainer = ModelTrainer(model, mode)
    trainer.train_model(True, False)


if __name__ == "__main__":
    test()
