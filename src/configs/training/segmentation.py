from collections import defaultdict

import keras


def constant_factory(value):
    return lambda: value


LEARNING_RATE = 0.0001
EPOCHS = 10
_compile_configs = defaultdict(
    constant_factory({"loss": "categorical_crossentropy", "metrics": ["accuracy"]}),
    {
        1: {"loss": "binary_crossentropy", "metrics": ["accuracy"]},
        2: {"loss": "binary_crossentropy", "metrics": ["accuracy"]},
    },
)

OPTIMIZER = keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    amsgrad=False,
)  # ,decay=1e-6)
