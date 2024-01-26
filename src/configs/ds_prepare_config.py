import numpy as np

from src.configs import model_config


TEST_SIZE = 0.2
VALIDATION_TRAIN_SIZE = 0.2
RANDOM_STATE = 0
RNG = np.random.RandomState(RANDOM_STATE)
BATCH_SIZE = 32
DS_SHUFFLE_BUFF_SIZE = BATCH_SIZE * 8
TARGET_SIZE = model_config.TARGET_SHAPE[0:2]
