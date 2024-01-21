from pathlib import Path

import numpy as np
from skimage.io import imread
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, TensorBoard
from configs import io_config, model_config, training_config, ds_prepare_config
from configs.training_config import COMPILE_CONFIGS
from load_ds import get_train_ds, get_val_ds

from utils.io_utils import save_model


def get_image_shapes(dir: Path):
    return imread(next(dir.iterdir())).shape


def train_model(model_name):
    with open(io_config.MODEL_SAVE_DIR / f"{model_name}_architecture.json") as f:
        json_model = f.read()
    model: keras.models.Model = model_from_json(json_model)
    # image_shape = get_images_shapes(io_config.TRAIN_IMAGES_DIR)

    train_ds = get_train_ds()
    val_ds = get_val_ds()

    optimizer = keras.optimizers.Adam(
        learning_rate=training_config.LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        amsgrad=False,
    )  # ,decay=1e-6)

    comp_config = COMPILE_CONFIGS[model_config.OUT_SIZE]
    comp_config["optimizer"] = optimizer
    comp_config["run_eagerly"] = training_config.DEBUG_MODEL
    model.compile(**comp_config)

    checkpointer = ModelCheckpoint(
        filepath=io_config.CHECKPOINTS_SAVE_DIR / f"{model_name}_{{epoch}}.keras",
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


if __name__ == "__main__":
    # pass
    # save_dataset_npy('carvana')
    save_model("unet0")
    train_model("unet0")
    # test_load_images_masks()
