from matplotlib import pyplot as plt
import numpy as np
from skimage.io import imread, imshow, show
import keras

from models import load_model, load_model_arch
from load_ds import get_dataset, get_test_ds


def main(model_name):
    # model = load_model(model_name)
    test_ds = get_test_ds()

    # preds = model.predict(test_ds)
    preds = np.load("data/predictions.npy")
    preds = (preds > 0.5).astype("uint8")
    images, masks = tuple(
        [list(el) for el in zip(*test_ds.unbatch().as_numpy_iterator())]
    )
    masks = np.array(masks)
    masks = (masks > 0.5).astype("uint8")

    mean_iou_metric = keras.metrics.MeanIoU(num_classes=2)
    mean_iou_metric.update_state(masks, preds)
    print(mean_iou_metric.result())

    preds = preds * 255
    masks = masks * 255

    def display(number):
        plt.figure()
        imshow(preds[number])
        show()
        plt.figure()
        imshow(masks[number])
        show()
        plt.figure()
        imshow(images[number])
        show()

    display(100)

    pass


if __name__ == "__main__":
    model_name = "unet0 batch_norm"
    main(model_name)
