import numpy as np
from PIL import Image
from collections import defaultdict


def to_binary_flat(image: Image.Image) -> np.ndarray:
    return (np.array(image).flatten() > 128).astype(np.uint8)


def to_image(arr: np.ndarray) -> Image.Image:
    x = np.expand_dims(arr, axis=0).reshape(28, 28) * 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def get_batches(samples: np.ndarray, labels: np.ndarray) -> np.ndarray:
    label_to_images = defaultdict(lambda: [])
    for image, label in zip(samples, labels):
        label_to_images[label].append(image)

    min_count = min(len(x) for x in label_to_images.values())
    batches = []
    for i in range(min_count):
        batches.append([label_to_images[label][i] for label in label_to_images])

    return np.array(batches)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)
