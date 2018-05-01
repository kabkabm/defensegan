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

"""The main class for training GANs."""

import argparse
import sys

import tensorflow as tf

from models.gan import MnistDefenseGAN, FmnistDefenseDefenseGAN, \
    CelebADefenseGAN
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


def main(cfg, *args):
    FLAGS = tf.app.flags.FLAGS
    ds_gan = {
        'mnist': MnistDefenseGAN, 'f-mnist': FmnistDefenseDefenseGAN,
        'celeba': CelebADefenseGAN,
    }
    GAN = ds_gan[FLAGS.dataset_name]

    gan = GAN(cfg=cfg, test_mode=not FLAGS.is_train)

    if FLAGS.is_train:
        gan.train()

    if FLAGS.train_encoder:
        gan.load(checkpoint_dir=FLAGS.init_path)
        gan.train(phase='just_enc')

    if FLAGS.save_recs:
        gan.reconstruct_dataset(ckpt_path=FLAGS.init_path,
                                max_num=FLAGS.max_num)

    if FLAGS.test_generator:
        gan.load_generator(ckpt_path=FLAGS.init_path)
        gan.sess.run(gan.global_step.initializer)
        gan.generate_image(iteration=0)

    if FLAGS.test_batch:
        gan.test_batch()

    if FLAGS.save_ds:
        gan.save_ds()


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python train.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_boolean("is_train", False,
                         "True for training, False for testing. [False]")
    flags.DEFINE_boolean("save_recs", False,
                         "True for saving reconstructions. [False]")
    flags.DEFINE_boolean("debug", False,
                         "True for debug. [False]")
    flags.DEFINE_boolean("test_generator", False,
                         "True for generator samples. [False]")
    flags.DEFINE_boolean("test_decoder", False,
                         "True for decoder samples. [False]")
    flags.DEFINE_boolean("test_again", False,
                         "True for not using cache. [False]")
    flags.DEFINE_boolean("test_batch", False,
                         "True for visualizing the batches and labels. [False]")
    flags.DEFINE_boolean("save_ds", False,
                         "True for saving the dataset in a pickle file. ["
                         "False]")
    flags.DEFINE_boolean("tensorboard_log", True, "True for saving "
                                                  "tensorboard logs. [True]")
    flags.DEFINE_boolean("train_encoder", False,
                         "Add an encoder to a pretrained model. ["
                         "False]")
    flags.DEFINE_boolean("init_with_enc", False,
                         "Initializes the z with an encoder, must run "
                         "--train_encoder first. [False]")
    flags.DEFINE_integer("max_num", -1,
                         "True for saving the dataset in a pickle file ["
                         "False]")
    flags.DEFINE_string("init_path", None, "Checkpoint path. [None]")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
