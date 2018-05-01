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

"""Contains the abstract class for models."""

import os

import tensorflow as tf
import yaml
from utils.misc import ensure_dir
from tensorflow.contrib import slim

from utils.dummy import DummySummaryWriter


class AbstractModel(object):
    def __init__(self, default_properties, test_mode=False, verbose=True,
                 cfg=None, **args):
        """The abstract model that the other models extend.

        Args:
            default_properties: The attributes of an experiment, read from a
            config file
            test_mode: If in the test mode, computation graph for loss will
            not be constructed, config will be saved in the output directory
            verbose: If true, prints debug information
            cfg: Config dictionary
            args: The rest of the arguments which can become object attributes
        """

        # Set attributes either from FLAGS or **args.
        self.cfg = cfg

        # Active session parameter.
        self.active_sess = None

        # Object attributes.
        default_properties.extend(
            ['tensorboard_log', 'output_dir', 'num_gpus'])
        self.default_properties = default_properties
        self.initialized = False
        self.verbose = verbose
        self.output_dir = 'output'

        local_vals = locals()
        args.update(local_vals)
        for attr in default_properties:
            if attr in args.keys():
                self._set_attr(attr, args[attr])
            else:
                self._set_attr(attr, None)

        # Runtime attributes.
        self.saver = None
        self.global_step = tf.train.get_or_create_global_step()
        self.global_step_inc = \
            tf.assign(self.global_step, tf.add(self.global_step, 1))

        # Phase: 1 train 0 test.
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.save_vars = {}
        self.save_var_prefixes = []
        self.dataset = None
        self.test_mode = test_mode

        self._set_checkpoint_dir()
        self._build()
        if not test_mode:
            self._save_cfg_in_ckpt()
            self._loss()

        self._initialize_summary_writer()


    def _load_dataset(self):
        pass

    def _build(self):
        pass

    def _loss(self):
        pass

    def test(self, input):
        pass

    def train(self):
        pass

    def _verbose_print(self, message):
        """Handy verbose print function"""
        if self.verbose:
            print(message)

    def _save_cfg_in_ckpt(self):
        """Saves the configuration in the experiment's output directory."""
        final_cfg = {}
        if hasattr(self, 'cfg'):
            for k in self.cfg.keys():
                if hasattr(self, k.lower()):
                    if getattr(self, k.lower()) is not None:
                        final_cfg[k] = getattr(self, k.lower())
            if not self.test_mode:
                with open(os.path.join(self.checkpoint_dir, 'cfg.yml'),
                          'w') as f:
                    yaml.dump(final_cfg, f)

    def _set_attr(self, attr_name, val):
        """Sets an object attribute from FLAGS if it exists, if not it
        prints out an error. Note that FLAGS is set from config and command
        line inputs.


        Args:
            attr_name: The name of the field.
            val: The value, if None it will set it from tf.apps.flags.FLAGS
        """

        FLAGS = tf.app.flags.FLAGS

        if val is None:
            if hasattr(FLAGS, attr_name):
                val = getattr(FLAGS, attr_name)
            elif hasattr(self, 'cfg'):
                if attr_name.upper() in self.cfg.keys():
                    val = self.cfg[attr_name.upper()]
                elif attr_name.lower() in self.cfg.keys():
                    val = self.cfg[attr_name.lower()]
        if val is None and self.verbose:
            print(
                '[-] {}.{} is not set.'.format(type(self).__name__, attr_name))

        setattr(self, attr_name, val)
        if self.verbose:
            print('[#] {}.{} is set to {}.'.format(type(self).__name__,
                                                   attr_name, val))

    def imsave_transform(self, imgs):
        return imgs

    def get_learning_rate(self, init_lr=None, decay_epoch=None,
                          decay_mult=None, iters_per_epoch=None,
                          decay_iter=None,
                          global_step=None, decay_lr=True):
        """Prepares the learning rate.
        
        Args:
            init_lr: The initial learning rate
            decay_epoch: The epoch of decay
            decay_mult: The decay factor
            iters_per_epoch: Number of iterations per epoch
            decay_iter: The iteration of decay [either this or decay_epoch
            should be set]
            global_step: 
            decay_lr: 

        Returns:
            `tf.Tensor` of the learning rate.
        """
        if init_lr is None:
            init_lr = self.learning_rate
        if global_step is None:
            global_step = self.global_step

        if decay_epoch:
            assert iters_per_epoch

            if iters_per_epoch is None:
                iters_per_epoch = self.iters_per_epoch
        else:
            assert decay_iter

        if decay_lr:
            if decay_epoch:
                decay_iter = decay_epoch * iters_per_epoch
            return tf.train.exponential_decay(init_lr,
                                              global_step,
                                              decay_iter,
                                              decay_mult,
                                              staircase=True)
        else:
            return tf.constant(self.learning_rate)


    def _set_checkpoint_dir(self):
        """Sets the directory containing snapshots of the model."""

        self.cfg_file = self.cfg['cfg_path']
        if 'cfg.yml' in self.cfg_file:
            ckpt_dir = os.path.dirname(self.cfg_file)

        else:
            ckpt_dir = os.path.join(self.output_dir,
                                    self.cfg_file.replace('experiments/cfgs/',
                                                          '').replace(
                                        'cfg.yml', '').replace(
                                        '.yml', ''))
            if not self.test_mode:
                postfix = ''
                ignore_list = ['dataset', 'cfg_file', 'batch_size']
                if hasattr(self, 'cfg'):
                    if self.cfg is not None:
                        for prop in self.default_properties:
                            if prop in ignore_list:
                                continue

                            if prop.upper() in self.cfg.keys():
                                self_val = getattr(self, prop)
                                if self_val is not None:
                                    if getattr(self, prop) != self.cfg[
                                        prop.upper()]:
                                        postfix += '-{}={}'.format(
                                            prop, self_val).replace('.', '_')

                ckpt_dir += postfix
            ensure_dir(ckpt_dir)

        self.checkpoint_dir = ckpt_dir
        self.debug_dir = self.checkpoint_dir.replace('output', 'debug')
        ensure_dir(self.debug_dir)

    def _initialize_summary_writer(self):
        # Setup the summary writer.
        if not self.tensorboard_log:
            self.summary_writer = DummySummaryWriter()
        else:
            sum_dir = os.path.join(self.checkpoint_dir, 'tb_logs')
            if not os.path.exists(sum_dir):
                os.makedirs(sum_dir)

            self.summary_writer = tf.summary.FileWriter(sum_dir)

    def _initialize_saver(self, prefixes=None, force=False, max_to_keep=5):
        """Initializes the saver object.

        Args:
            prefixes: The prefixes that the saver should take care of.
            force (optional): Even if saver is set, reconstruct the saver
                object.
            max_to_keep (optional):
        """
        if self.saver is not None and not force:
            return
        else:
            if prefixes is None or not (
                type(prefixes) != list or type(prefixes) != tuple):
                raise ValueError(
                    'Prefix of variables that needs saving are not defined')

            prefixes_str = ''
            for pref in prefixes:
                prefixes_str = prefixes_str + pref + ' '

            print('[#] Initializing it with variable prefixes: {}'.format(
                prefixes_str))
            saved_vars = []
            for pref in prefixes:
                saved_vars.extend(slim.get_variables(pref))

            self.saver = tf.train.Saver(saved_vars, max_to_keep=max_to_keep)

    def set_session(self, sess):
        """"""
        if self.active_sess is None:
            self.active_sess = sess
        else:
            raise EnvironmentError("Session is already set.")

    @property
    def sess(self):
        if self.active_sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.active_sess = tf.Session(config=config)

        return self.active_sess

    def close_session(self):
        if self.active_sess:
            self.active_sess.close()

    def load(self, checkpoint_dir=None, prefixes=None, saver=None):
        """Loads the saved weights to the model from the checkpoint directory
        
        Args:
            checkpoint_dir: The path to saved models
        """
        if prefixes is None:
            prefixes = self.save_var_prefixes
        if self.saver is None:
            print('[!] Saver is not initialized')
            self._initialize_saver(prefixes=prefixes)

        if saver is None:
            saver = self.saver

        if checkpoint_dir is None:
            checkpoint_dir = self.checkpoint_dir

        if not os.path.isdir(checkpoint_dir):
            try:
                saver.restore(self.sess, checkpoint_dir)
            except:
                print(" [!] Failed to find a checkpoint at {}".format(
                    checkpoint_dir))
        else:
            print(" [-] Reading checkpoints... {} ".format(checkpoint_dir))

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(self.sess,
                              os.path.join(checkpoint_dir, ckpt_name))
            else:
                print(
                    " [!] Failed to find a checkpoint "
                    "within directory {}".format(checkpoint_dir))
                return False

        print(" [*] Checkpoint is read successfully from {}".format(
            checkpoint_dir))

        return True

    def add_save_vars(self, prefixes):
        """Prepares the list of variables that should be saved based on
        their name prefix.

        Args:
            prefixes: Variable name prefixes to find and save.
        """

        for pre in prefixes:
            pre_vars = slim.get_variables(pre)
            self.save_vars.update(pre_vars)

        var_list = ''
        for var in self.save_vars:
            var_list = var_list + var.name + ' '

        print ('Saving these variables: {}'.format(var_list))

    def input_transform(self, images):
        pass

    def input_pl_transform(self):
        self.real_data = self.input_transform(self.real_data_pl)
        self.real_data_test = self.input_transform(self.real_data_test_pl)

    def initialize_uninitialized(self, ):
        """Only initializes the variables of a TensorFlow session that were not
        already initialized.
        """
        # List all global variables.
        sess = self.sess
        global_vars = tf.global_variables()

        # Find initialized status for all variables.
        is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
        is_initialized = sess.run(is_var_init)

        # List all variables that were not previously initialized.
        not_initialized_vars = [var for (var, init) in
                                zip(global_vars, is_initialized) if not init]
        for v in not_initialized_vars:
            print('[!] not init: {}'.format(v.name))
        # Initialize all uninitialized variables found, if any.
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

    def save(self, prefixes=None, global_step=None, checkpoint_dir=None):
        if global_step is None:
            global_step = self.global_step
        if checkpoint_dir is None:
            checkpoint_dir = self._set_checkpoint_dir

        ensure_dir(checkpoint_dir)
        self._initialize_saver(prefixes)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_save_name),
                        global_step=global_step)
        print('Saved at iter {} to {}'.format(self.sess.run(global_step),
                                              checkpoint_dir))

    def initialize(self, dir):
        self.load(dir)
        self.initialized = True
