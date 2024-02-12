from abc import ABC, abstractmethod
from pathlib import Path

from configs import io_config


class ModelManager:
    _model_name_sep = "_"
    _save_name = "model.keras"
    _arch_name = "architecture.json"
    _summary_name = "summary.txt"
    _plot_name = "plot.png"

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def __init__(self, model_name: str, model_mode: str) -> None:
        self._model = None
        self._model_name = model_name
        self._model_mode = model_mode
        super().__init__()

    def save_model(self):
        pass

    def load_model(self):
        pass

    def read_model(self):
        pass

    def write_model(self):
        pass

    def _model_dir(self):
        return io_config.MODEL_SAVE_DIR / Path(
            self._model_name_sep.join([self._model_name, self._model_mode])
        )
