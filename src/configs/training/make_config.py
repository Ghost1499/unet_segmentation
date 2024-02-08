from configparser import ConfigParser, ExtendedInterpolation
import json
from pprint import pprint

from keras.optimizers import Adam

from src.configs import ds_prepare_config


__optimizers = {"adam": Adam}
__train_config_path = "configs/train.ini"


def _create_oprimizer(name, **kwargs):
    try:
        opt = __optimizers[name](**kwargs)
        return opt
    except KeyError as err:
        raise Exception("Некорректное имя оптимизатора.", *err.args)


def make_config(is_debug, mode):
    train_config = ConfigParser(interpolation=ExtendedInterpolation())
    read_files = train_config.read(__train_config_path)
    if not read_files:
        raise Exception("Конфигурация обучения модели не прочитана.")

    if mode not in train_config.sections():
        raise ValueError("Некорректный режим работы.")

    try:
        compile_params = {}
        compile_params["optimizer"] = _create_oprimizer(
            train_config[mode]["optimizer"],
            learning_rate=train_config[mode].getfloat("learning_rate"),
        )
        compile_params["loss"] = train_config[mode]["loss"]
        compile_params["metrics"] = json.loads(train_config[mode]["metrics"])
        compile_params["run_eagerly"] = is_debug
        train_params = {
            "batch_size": ds_prepare_config.BATCH_SIZE,
            "epochs": train_config[mode].getint("epochs"),
            "validation_split": ds_prepare_config.VALIDATION_TRAIN_SIZE,
            "compile_params": compile_params,
        }
    except KeyError as err:
        raise Exception("Параметр конфигурации обучения не найден", *err.args)
    return train_params


def test():
    pprint(make_config(False, "segmentation"))


if __name__ == "__main__":
    test()
