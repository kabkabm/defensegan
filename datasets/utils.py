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

"""Data handling related utilities live in this module."""

import tensorflow as tf

from datasets.celeba import CelebA
from datasets.fmnist import FMnist
from datasets.mnist import Mnist


def create_generator(dataset_name, split, batch_size, randomize,
                     attribute=None):
    """Creates a batch generator for the dataset.

    Args:
        dataset_name: `str`. The name of the dataset.
        split: `str`. The split of data. It can be `train`, `val`, or `test`.
        batch_size: An integer. The batch size.
        randomize: `bool`. Whether to randomize the order of images before
            batching.
        attribute (optional): For cele

    Returns:
        image_batch: A Python generator for the images.
        label_batch: A Python generator for the labels.
    """
    flags = tf.app.flags.FLAGS

    if dataset_name.lower() == 'mnist':
        ds = Mnist()
    elif dataset_name.lower() == 'f-mnist':
        ds = FMnist()
    elif dataset_name.lower() == 'celeba':
        ds = CelebA(attribute=attribute)
    else:
        raise ValueError("Dataset {} is not supported.".format(dataset_name))

    ds.load(split=split, randomize=randomize)

    def get_gen():
        for i in range(0, len(ds) - batch_size, batch_size):
            image_batch, label_batch = ds.images[
                                       i:i + batch_size], \
                                       ds.labels[i:i + batch_size]
            yield image_batch, label_batch

    return get_gen


def get_generators(dataset_name, batch_size, randomize=True, attribute='gender'):
    """Creates batch generators for datasets.

    Args:
        dataset_name: A `string`. Name of the dataset.
        batch_size: An `integer`. The size of each batch.
        randomize: A `boolean`.
        attribute: A `string`. If the dataset name is `celeba`, this will
         indicate the attribute name that labels should be returned for.

    Returns:
        Training, validation, and test dataset generators which are the
            return values of `create_generator`.
    """
    splits = ['train', 'val', 'test']
    gens = []
    for i in range(3):
        if i > 0:
            randomize = False
        gens.append(
            create_generator(dataset_name, splits[i], batch_size, randomize,
                             attribute=attribute))

    return gens