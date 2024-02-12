from abc import ABC, abstractmethod


class ModelManager:
    def __init__(self, model_name) -> None:
        super().__init__()

    def save_model(self):
        pass

    def load_model(self):
        pass

    def read_model(self):
        pass

    def write_model(self):
        pass
