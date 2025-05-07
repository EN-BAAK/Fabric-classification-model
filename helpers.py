from typing import Optional, Tuple, Union
from math import floor
import matplotlib.pyplot as plt
import tensorflow as tf

def split_data_set(
    dataset,
    training_percent: float = 0.8,
    validation_percent: Optional[float] = None,
    shuffle: bool = True,
    shuffle_size: int = 10000,
    seed: int = 200
) -> Union[Tuple, Tuple[object, object]]:
    total_batches = len(dataset)
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed = seed)

    training_batches = floor(total_batches * training_percent)
    skip = 0

    train_ds = dataset.skip(skip).take(training_batches)
    skip += training_batches

    validate_ds = None
    if validation_percent is not None:
        validation_batches = floor(total_batches * validation_percent)
        validate_ds = dataset.skip(skip).take(validation_batches)
        skip += validation_batches

    test_ds = dataset.skip(skip)

    if validate_ds is not None:
        return train_ds, validate_ds, test_ds
    else:
        return train_ds, test_ds


def view_dataset_batches(dataset, ar_classes, take=1, no_imgs=12, figsize=10, rows=3, columns=4):
    plt.figure(figsize=(figsize, figsize))

    for image_batch, labels_batch in dataset.take(take):
        for i in range(min(no_imgs, len(image_batch))):
            plt.subplot(rows, columns, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))

            label_index = labels_batch[i].numpy()
            
            plt.title(ar_classes[label_index])
            plt.axis("off")

    plt.tight_layout()
    plt.show()

def convert_to_gray_scale(img, label):
    img = tf.image.rgb_to_grayscale(img)
    return img, label