import math
import numpy
import os
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import random

from chainer.dataset import dataset_mixin


class CroppingImageDatasetIO(dataset_mixin.DatasetMixin):
    def __init__(self, pairs, root, mean, crop_h, crop_w, random=True,
                 image_dtype=numpy.float32, label_dtype=numpy.int32):
        self._pairs = pairs
        self._root = root
        self._mean = mean.astype('f')
        self._crop_h = crop_h
        self._crop_w = crop_w
        self._random = random
        self._image_dtype = image_dtype
        self._label_dtype = label_dtype

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        crop_h = self._crop_h
        crop_w = self._crop_w
        path, label = self._pairs[i]
        path = os.path.join(self._root, path)
        image = _read_image(path, crop_h, crop_w, self._image_dtype)
        mean = self._mean
        image = _transform_image(image, crop_h, crop_w, mean, self._random)
        return image, numpy.array(label, dtype=self._label_dtype)


def _read_image(path, crop_h, crop_w, dtype):
    im = Image.open(path)

    w, h = im.size
    if w < crop_w or h < crop_h:
        if w < h:
            h = math.ceil(h * crop_w / w)
            w = crop_w
        else:
            w = math.ceil(w * crop_h / h)
            h = crop_h
        im = im.resize((w, h))
    try:
        image = numpy.asarray(im, dtype=dtype)
    finally:
        im.close()
        del im
    return image


def _transform_image(image, crop_h, crop_w, mean, crop_random=False):
    if image.ndim == 2:
        # image is grayscale
        image = image[:, :, numpy.newaxis]
    image = image[:, :, :3]  # Remove alpha (i.e. transparency)
    image = image.transpose(2, 0, 1)  # (h, w, c) -> (c, h, w)
    _, h, w = image.shape

    if crop_random:
        # Randomly crop a region and flip the image
        top = random.randint(0, max(h - crop_h - 1, 0))
        left = random.randint(0, max(w - crop_w - 1, 0))
        if random.randint(0, 1):
            image = image[:, :, ::-1]
    else:
        # Crop the center
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
    bottom = top + crop_h
    right = left + crop_w

    image = image[:, top:bottom, left:right]
    image = image - mean[:, :crop_h, :crop_w]  # Only use top left of mean
    image *= (1.0 / 255.0)  # Scale to [0, 1]
    return image


def read_pairs(path):
    """Read path to image and label pairs from file.

    Args:
        path (str): Path to the image-label pair file.

    Returns:
        list of pairs: Each pair type is ``(str, int)``. Which first element is
            a path to image and second element is a label.
    """
    pairs = []
    with open(path) as f:
        for line in f:
            path, label = line.split()
            label = int(label)
            pairs.append((path, label))
    return pairs
