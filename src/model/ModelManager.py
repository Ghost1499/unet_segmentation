from abc import ABC, abstractmethod
from pathlib import Path

import keras.saving

from configs import io_config


class ModelManager:
    _model_name_sep = "_"
    _save_name = "model.keras"
    _arch_name = "architecture.json"
    _summary_name = "summary.txt"
    _plot_name = "plot.png"

    def __init__(self, model_name: str, model_mode: str) -> None:
        self._model_name = model_name
        self._model_mode = model_mode
        super().__init__()

    def save_model(self,model:keras.Model,overwrite=False):
        save_path = self._model_dir()/self._save_name
        if not overwrite and save_path.exists():
            raise Exception("Модель по заданному пути уже существует.",save_path)
        model.save(save_path)

    def load_model(self):
        self.model = keras.saving.load_model(self._model_dir() / self._save_name)

    def read_model(self):
        if self.

    def write_model(self):
        pass
        
    def _model_dir(self):
        return io_config.MODEL_SAVE_DIR / Path(
            self._model_name_sep.join([self._model_name, self._model_mode])
        )
