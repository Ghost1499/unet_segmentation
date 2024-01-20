from collections import defaultdict


def constant_factory(value):
    return lambda: value


LEARNING_RATE = 0.0001
EPOCHS = 10
COMPILE_CONFIGS = defaultdict(
    constant_factory({"loss": "categorical_crossentropy", "metrics": ["accuracy"]}),
    {
        1: {"loss": "binary_crossentropy", "metrics": ["accuracy"]},
        2: {"loss": "binary_crossentropy", "metrics": ["accuracy"]},
    },
)
DEBUG_MODEL = True
