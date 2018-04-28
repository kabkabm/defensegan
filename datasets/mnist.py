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

"""Contains the class for handling the MNIST dataset."""

import os

import numpy as np

from datasets.dataset import Dataset


class Mnist(Dataset):
    """Implements the Dataset class to handle MNIST.

    Attributes:
        y_dim: The dimension of label vectors (number of classes).
        split_data: A dictionary of
            {
                'train': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'val': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'test': Images of np.ndarray, Int array of labels, and int
                array of ids.
            }
    """

    def __init__(self):
        super(Mnist, self).__init__('mnist')
        self.y_dim = 10
        self.split_data = {}

    def load(self, split='train', lazy=True, randomize=True):
        """Implements the load function.

        Args:
            split: Dataset split, can be [train|dev|test], default: train.
            lazy: Not used for MNIST.

        Returns:
             Images of np.ndarray, Int array of labels, and int array of ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """

        if split in self.split_data.keys():
            return self.split_data[split]

        data_dir = self.data_dir

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_labels = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000)).astype(np.float)

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        if split == 'train':
            images = train_images[:50000]
            labels = train_labels[:50000]
        elif split == 'val':
            images = train_images[50000:60000]
            labels = train_labels[50000:60000]
        elif split == 'test':
            images = test_images
            labels = test_labels

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        images = np.reshape(images, [-1, 28, 28, 1])
        self.split_data[split] = [images, labels]
        self.images = images
        self.labels = labels

        return images, labels
