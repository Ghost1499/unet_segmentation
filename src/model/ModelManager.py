from abc import ABC, abstractmethod
from json import load
from pathlib import Path

import keras.saving
from keras.models import model_from_json

from configs import io_config


class ModelManager:
    _model_name_sep = "_"
    _save_fname = "model.keras"
    _arch_fname = "architecture.json"
    _summary_fname = "summary.txt"
    _plot_fname = "plot.png"

    def __init__(self, model_name: str, model_mode: str) -> None:
        self._model_name = model_name
        self._model_mode = model_mode
        super().__init__()

    def load_model(self):
        load_path = self._model_dir() / self._save_fname
        if not load_path.exists():
            raise Exception("Заданный путь загрузки модели не существует.")
        return keras.saving.load_model(load_path)

    def save_model(self, model: keras.Model, overwrite=False):
        save_path = self._model_dir() / self._save_fname
        if not overwrite and save_path.exists():
            raise Exception(
                "Сохранённая модель по заданному пути уже существует.", save_path
            )
        model.save(save_path)

    def read_model(self):
        read_path = self._model_dir() / self._arch_fname
        if not read_path.exists():
            raise Exception(
                "Заданный путь чтения архитектуры модели не существует.", read_path
            )
        with open(read_path) as json_file:
            json_model = json_file.read()
        return model_from_json(json_model)

    def write_model(self):
        pass

    def _model_dir(self):
        return io_config.MODEL_SAVE_DIR / Path(
            self._model_name_sep.join([self._model_name, self._model_mode])
        )
