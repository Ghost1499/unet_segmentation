from keras import backend as K

def gather_channels(*xs):
    """
    Преобразование данных в другую форму (разворачивает каналы)
    """
    return xs


def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x