from abc import ABC, abstractmethod


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

    def __init__(self, model_name: str, mode: str) -> None:
        self._model = None
        super().__init__()

    def save_model(self):
        pass

    def load_model(self):
        pass

    def read_model(self):
        pass

    def write_model(self):
        pass
