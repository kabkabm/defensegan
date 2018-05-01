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
# ==============================================================================

"""Contains the class for handling the CelebA dataset."""

import os

import numpy as np

from datasets.dataset import Dataset, LazyDataset


class CelebA(Dataset):
    """CelebA class implementing Dataset."""

    def __init__(self, center_crop_size=108, resize_size=64, attribute=None):
        """CelebA constructor.

        Args:
            center_crop_size: An integer defining the center crop square
            dimensions.
            resize_size: An integer for the final size of the cropped image.
            attribute: A string which is the attribute name according to the
                CelebA's label file header.
        """

        super(CelebA, self).__init__('celebA')
        self.y_dim = 0
        self.split_data = {}
        self.image_size = center_crop_size
        self.resize_size = resize_size
        # The attribute represents which attribute to use in case of
        # classification.
        self.attribute = attribute
        # Only gender classification is supported.
        self.attr_dict = {'gender': ['male']}

    def load(self, split='train', lazy=True, randomize=False):
        """Loads the dataset according to split.

        Args:
            split: A string [train|val|test] referring to the dataset split.
            lazy (optional): If True, only loads the file paths and creates a
                LazyDataset object (default True).
            randomize (optional): `True` will randomize the data.

        Returns:
            A LazyDataset (if lazy is True) or a numpy array containing all
            the images, labels, and image ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """

        attribute = self.attribute

        # If split data has already been loaded, return it.
        if split in self.split_data.keys():
            return self.split_data[split]

        # Start and end indices of different CelebA splits.
        if split == 'train':
            start = 1
            end = 162770
        elif split == 'val':
            start = 162771
            end = 182637
        elif split == 'test':
            start = 182638
            end = 202599
        else:
            raise ValueError('[!] Invalid split {}.'.format(split))

        data_dir = self.data_dir

        # Lazy dataset loading.
        fps = [os.path.join(data_dir, '{:06d}.jpg'.format(i)) for i in
               range(start, end + 1)]

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(fps)
            np.random.set_state(rng_state)

        images = LazyDataset(fps, self.image_size, self.resize_size)
        # Access images if not lazy.
        if not lazy:
            images = images[:len(images)]

        if attribute is None:  # No class information needed.
            labels = None  # Labels set to None.
            ids = np.array(range(0, end - start + 1),
                           dtype=int)  # All indices are valid.
        else:
            # If attribute is valid.
            if self.attr_dict.has_key(attribute):
                # Get list of classes to consider.
                attr_list = self.attr_dict[attribute]
                with open(os.path.join(self.data_dir, 'list_attr_celeba.txt'),
                          'r') as f:
                    flines = f.readlines()
                    class_names = [s.lower().replace(' ', '_') for s in
                                   flines[1].strip().split()]
                    # Get indices of relevant columns.
                    cols = [i for i, x in enumerate(class_names) if
                            x in attr_list]
                    cols = np.asarray(cols, dtype=int)
                    face_attributes = [[int(x) for x in l.split()[1:]] for l in
                                       [ll.strip() for ll in flines[2:]]]
                    face_attributes = (np.asarray(face_attributes,
                                                  dtype=int) + 1) // 2
                    face_attributes = face_attributes[start - 1:end, cols]
                    labels = face_attributes.reshape(-1)
            else:
                raise ValueError(
                    '[!] Invalid attribute {} for CelebA dataset.'.format(
                        attribute))

        self.split_data[split] = [images, labels]

        self.images = images
        self.labels = labels

        return images, labels
