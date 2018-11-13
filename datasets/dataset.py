# Copyright 2018 The Defense-GAN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Contains the classes:
 Dataset: All datasets used in the project implement this class.
 LazyDataset: A class for loading data in a lazy manner from file paths.
 LazyPickledDataset: A class for loading pickled data from filepaths.

defined here."""

import cPickle
import os

import numpy as np
import scipy
import scipy.misc


class Dataset(object):
    """The abstract class for handling datasets.

    Attributes:
        name: Name of the dataset.
        data_dir: The directory where the dataset resides.
    """

    def __init__(self, name, data_dir='./data'):
        """The datasaet default constructor.

            Args:
                name: A string, name of the dataset.
                data_dir (optional): The path of the datasets on disk.
        """

        self.data_dir = os.path.join(data_dir, name)
        self.name = name
        self.images = None
        self.labels = None

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.images)

    def load(self, split):
        """ Abstract function specific to each dataset."""
        pass


class LazyDataset(object):
    """The Lazy Dataset class.
    Instead of loading the whole dataset into memory, this class loads
    images only when their index is accessed.

        Attributes:
            fps: String list of file paths.
            center_crop_dim: An integer for the size of center crop (after
                loading the images).
            resize_size: The final resize size (after loading the images).
    """

    def __init__(self, filepaths, center_crop_dim, resize_size,
                 transform_type=None):
        """LazyDataset constructor.

        Args:
            filepaths: File paths.
            center_crop_dim: The dimension of the center cropped square.
            resize_size: Final size to resize the center crop of the images.
        """

        self.filepaths = filepaths
        self.center_crop_dim = center_crop_dim
        self.resize_size = resize_size
        self.transform_type = transform_type

    def _get_image(self, image_path):
        """Retrieves an image at a given path and resizes it to the
        specified size.

        Args:
            image_path: Path to image.

        Returns:
            Loaded and transformed image.
        """

        # Read image at image_path.
        image = scipy.misc.imread(image_path).astype(np.float)

        # Return transformed image.
        return _prepare_image(image, self.center_crop_dim,
                              self.center_crop_dim,
                              resize_height=self.resize_size,
                              resize_width=self.resize_size,
                              is_crop=True)

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.filepaths)

    def __getitem__(self, index):
        """Loads and returns images specified by index.

        Args:
            index: Indices of images to load.

        Returns:
            Loaded images.

        Raises:
            TypeError: If index is neither of: int, slice, np.ndarray.
        """

        # Case of a single integer index.
        if isinstance(index, int):
            return self._get_image(self.filepaths[index])
        # Case of a slice or array of indices.
        elif isinstance(index, slice):
            if isinstance(index, slice):
                if index.start is None:
                    index = range(index.stop)
                elif index.step is None:
                    index = range(index.start, index.stop)
                else:
                    index = range(index.start, index.stop, index.step)
            return np.array(
                [self._get_image(self.filepaths[i]) for i in index]
            )
        else:
            try:
                inds = [int(i) for i in index]
                return np.array(
                    [self._get_image(self.filepaths[i]) for i in inds]
                )
            except TypeError:
                raise TypeError("Index must be an integer, a slice, a container or an integer generator.")

    def get_subset(self, indices):
        """Gets a subset of the images

        Args:
            indices: The indices of the images that are needed. It's like
            lazy indexing without loading.

        Raises:
            TypeError if index is not a slice.
        """
        if isinstance(indices, int):
            self.filepaths = self.filepaths[indices]
        elif isinstance(indices, slice) or isinstance(indices, np.ndarray):
            self.filepaths = [self.filepaths[i] for i in indices]
        else:
            raise TypeError("Index must be an integer or a slice.")

    @property
    def shape(self):
        return tuple([None] + list(self._get_image(self.filepaths[0]).shape))

    @property
    def dtype(self):
        return self._get_image(self.filepaths[0]).dtype


class PickleLazyDataset(LazyDataset):
    """This dataset is a lazy dataset for working with saved pickle files
    (of typically generated images) on disk without loading them.
    """

    def __init__(self, filepaths, shape=None):
        """The constructor for instances of this class.

        Args:
            filepaths: List of strings. The list of file paths.
            shape (optional): Shape of the loaded images in case the images
                are saved as a vector.
        """
        self.filepaths = filepaths
        self.image_shape = shape

    def __len__(self):
        return len(self.filepaths)

    def _get_image(self, filepath):
        with open(filepath) as f:
            return cPickle.load(f).reshape(self.image_shape)

    @property
    def shape(self):
        im = self.__getitem__(0)
        return [len(self.filepaths)] + list(im.shape)


def _prepare_image(image, crop_height, crop_width, resize_height=64,
                   resize_width=64, is_crop=True):
    """Prepares an image by first applying an optional center
    crop, then resizing it.

    Args:
        image: Input image.
        crop_height: The height of the crop.
        crop_width: The width of the crop.
        resize_height: The resize height after cropping.
        resize_width: The resize width after cropping.
        is_crop: If True, first apply a center crop.

    Returns:
        The cropped and resized image.
    """

    def center_crop(image, crop_h, crop_w, resize_h=64, resize_w=64):
        """Performs a center crop followed by a resize.

        Args:
            image: Image of type np.ndarray
            crop_h: The height of the crop.
            crop_w: The width of the crop.
            resize_h: The resize height after cropping.
            resize_w: The resize width after cropping.

        Returns:
            The cropped and resized image of type np.ndarray.
        """
        if crop_w is None:
            crop_w = crop_h
        h, w = image.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        # Crop then resize.
        return scipy.misc.imresize(image[j:j + crop_h, i:i + crop_w],
                                   [resize_h, resize_w])

    # Optionally crop the image. Then resize it.
    if is_crop:
        cropped_image = center_crop(image, crop_height, crop_width,
                                    resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height,
                                                    resize_width])
    return cropped_image
