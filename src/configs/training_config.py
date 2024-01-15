from collections import defaultdict


def constant_factory(value):
    return lambda: value


LEARNING_RATE = 0.0001
EPOCHS = 32
COMPILE_CONFIGS = defaultdict(
    constant_factory({"loss": "categorical_crossentropy", "metrics": ["accuracy"]}),
    {
        1: {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
        2: {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]},
    },
)
DEBUG_MODEL = True
