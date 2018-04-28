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

"""Visualization utilities."""

import gc
import os

import numpy as np
import scipy.misc

from utils.misc import static_vars, make_dir


@static_vars(plt_counter=0)
def save_plot(plt, fname=None, save_dir='debug/plots/'):
    plt.tight_layout()
    plt.draw()
    if fname is None:
        fname = 'plot_{}.png'.format(save_plot.plt_counter)
        save_plot.plt_counter = save_plot.plt_counter + 1

    make_dir(save_dir)
    if not 'png' in fname and not 'pdf' in fname:
        fname = fname + '.png'

    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path)
    print('[-] Saved plot to {}'.format(save_path))
    plt.clf()
    plt.close()
    gc.collect()


def save_images_files(images, prefix='im', labels=None, output_dir=None,
                      postfix=''):
    if prefix is None and labels is None:
        prefix = '{}_image.png'
    else:
        prefix = prefix + '_{:03d}'
    if labels is not None:
        prefix = prefix + '_{:03d}'

    prefix = prefix + postfix + '.png'

    assert len(images.shape) == 4, 'images should be a 4D np array uint8'
    for i in range(images.shape[0]):
        image = images[i]
        if labels is None:
            save_image(image, fname=prefix.format(i), dir_path=output_dir)
        else:
            save_image(image, fname=prefix.format(i, int(labels[i])),
                       dir_path=output_dir)


@static_vars(image_counter=0)
def save_image(image, fname=None, dir_path='debug/images/'):
    if fname is None:
        fname = 'image_{}.png'.format(save_image.image_counter)
        save_image.image_counter = save_image.image_counter + 1
    make_dir(dir_path)
    fpath = os.path.join(dir_path, fname)
    save_image_core(image, fpath)


def save_image_core(image, path):
    """Save an image as a png file"""
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = image.transpose([1, 2, 0])
    image = ((image.squeeze() * 1.0 - image.min()) / (
        image.max() - image.min() + 1e-7)) * 255
    image = image.astype(np.uint8)
    scipy.misc.imsave(path, image)

    print('[#] saved image to: {}'.format(path))
