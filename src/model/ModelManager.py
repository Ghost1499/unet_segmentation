from abc import ABC, abstractmethod
from json import load
import json
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

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_mode(self):
        return self._model_mode

    @model_mode.setter
    def model_mode(self, value):
        self._model_mode = value

    def __init__(self, model_name: str, model_mode: str) -> None:
        self._model_name = model_name
        self._model_mode = model_mode
        super().__init__()

    def load_model(self):
        load_path = self._model_dir() / self._save_fname
        try:
            return keras.saving.load_model(load_path)
        except ValueError as err:
            raise Exception("Заданный путь загрузки модели не существует.", *err.args)

    def save_model(self, model: keras.Model, overwrite=False):
        save_path = self._model_dir() / self._save_fname
        if not overwrite and save_path.exists():
            raise Exception(
                "Сохранённая модель по заданному пути уже существует.", save_path
            )
        model.save(save_path, overwrite)

    def read_model(self):
        read_path = self._model_dir() / self._arch_fname
        try:
            with open(read_path, "r") as json_file:
                json_model = json_file.read()
            return model_from_json(json_model)
        except FileNotFoundError as err:
            raise Exception(
                "Заданный путь чтения архитектуры модели не существует.", *err.args
            )

    def write_model(self, model: keras.Model, overwrite=False):
        write_path = self._model_dir() / self._arch_fname
        try:
            with open(write_path, "x") as json_file:
                json_file.write(model.to_json())
        except FileExistsError as err:
            raise Exception(
                "Архитектура модели по заданному пути уже существует.", *err.args
            )

    def _model_dir(self):
        return io_config.MODEL_SAVE_DIR / Path(
            self._model_name_sep.join([self._model_name, self._model_mode])
        )
