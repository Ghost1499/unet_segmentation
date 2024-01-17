import numpy as np
from keras import backend as K
import tensorflow as tf

from utils.model_utils import gather_channels, round_if_needed


def F1(y_true, y_pred):
    pr = tf.keras.metrics.Precision()
    pr.update_state(y_true, y_pred)
    re = tf.keras.metrics.Recall()
    re.update_state(y_true, y_pred)
    pr_ = pr.result().numpy()
    re_ = re.result().numpy()
    f1_score = 2 * (pr_ * re_) / (pr_ + re_)
    return f1_score


def f1_metric(y_true_, y_pred_):
    y_true = np.array(y_true_)
    y_pred = np.array(y_pred_)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)


def precision(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    score = (tp + smooth) / (tp + fp + smooth)
    score = average(score, class_weights)
    return score


def recall(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    tp = K.sum(y_true * y_pred, axis=axes)
    fn = K.sum(y_true, axis=axes) - tp
    score = (tp + smooth) / (tp + fn + smooth)
    score = average(score, class_weights)
    return score


def f1_score(y_true, y_pred):
    pr = precision(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None)
    re = recall(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None)
    f1_score = 2 * (pr * re) / (pr + re)
    return f1_score


def iou_score(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None):
    # y_true = K.one_hot(K.squeeze(K.cast(y_true, tf.int32), axis=-1), n_classes)

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]

    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection

    score = (intersection + smooth) / (union + smooth)
    score = average(score, class_weights)

    return score


def Jaccard_Loss(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None):
    return 1 - iou_score(y_true, y_pred, class_weights=1.0, smooth=1e-5, threshold=None)


def tversky(y_true, y_pred, alpha=0.7, class_weights=1.0, smooth=1e-5, threshold=None):
    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]

    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = (tp + smooth) / (tp + alpha * fn + (1 - alpha) * fp + smooth)
    score = average(score, class_weights)
    return score


def tversky_loss(
    y_true, y_pred, alpha=0.7, class_weights=1.0, smooth=1e-5, threshold=None
):
    return 1 - tversky(
        y_true, y_pred, alpha=0.7, class_weights=1.0, smooth=1e-5, threshold=None
    )


def focal_tversky_loss(
    y_true, y_pred, alpha=0.7, gamma=1.25, lass_weights=1.0, smooth=1e-5, threshold=None
):
    return K.pow(
        1
        - tversky(
            y_true, y_pred, alpha=0.7, class_weights=1.0, smooth=1e-5, threshold=None
        ),
        gamma,
    )


def dice_coef(y_true, y_pred):  # type: ignore
    return (2.0 * K.sum(y_true * y_pred) + 1.0) / (K.sum(y_true) + K.sum(y_pred) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def main():
    y_true, y_pred = [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]
    s = F1(y_true, y_pred)

    # y_true=np.array([0, 0, 1, 1]);y_pred=np.array([1, 1, 1, 1])
    s = f1_metric(y_true, y_pred)


if __name__ == "__main__":
    main()
