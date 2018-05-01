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

"""Contains the GAN implementations of the abstract model class."""

import cPickle
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

import tflib
import tflib.cifar10
import tflib.mnist
import tflib.plot
import tflib.save_images
from datasets.utils import get_generators
from models.base_model import AbstractModel
from models.dataset_models import mnist_generator, celeba_discriminator, \
    mnist_discriminator, celeba_generator
from utils.misc import ensure_dir
from utils.visualize import save_images_files


class DefenseGANBase(AbstractModel):
    def __init__(self, cfg=None, test_mode=False, verbose=True, **args):
        default_attributes = ['dataset_name', 'batch_size', 'use_bn',
                              'test_batch_size',
                              'mode', 'gradient_penalty_lambda', 'train_iters',
                              'critic_iters', 'latent_dim', 'net_dim',
                              'input_transform_type',
                              'debug', 'rec_iters', 'image_dim', 'rec_rr',
                              'rec_lr', 'test_again', 'loss_type',
                              'attribute']

        self.dataset_name = None  # Name of the datsaet.
        self.batch_size = 32  # Batch size for training the GAN.
        self.use_bn = True  # Use batchnorm in the discriminator and generator.
        self.test_batch_size = 20  # Batch size for test time.
        self.mode = 'gp-wgan'  # The mode of training the GAN (default: gp-wgan).
        self.gradient_penalty_lambda = 10.0  # Gradient penalty scale.
        self.train_iters = 30000  # Number of training iterations.
        self.critic_iters = 5  # Critic iterations per training step.
        self.latent_dim = None  # The dimension of the latent vectors.
        self.net_dim = None  # The complexity of network per layer.
        self.input_transform_type = 0  # The normalization used for the inputs.
        self.debug = False  # Debug info will be printed.
        self.rec_iters = 200  # Number of reconstruction iterations.
        self.image_dim = [None, None, None]  # [height, width, number of channels] of the output image.
        self.rec_rr = 10  # Number of random restarts for the reconstruction

        self.rec_lr = 10.0  # The reconstruction learning rate.
        self.test_again = False  # If true, do not use the cached info for test phase.
        self.attribute = 'gender'

        # Should be implemented in the child classes.
        self.discriminator_fn = None
        self.generator_fn = None
        self.train_data_gen = None

        self.model_save_name = 'GAN.model'

        super(DefenseGANBase, self).__init__(default_attributes,
                                             test_mode=test_mode,
                                             verbose=verbose, cfg=cfg, **args)
        self.save_var_prefixes = ['Generator', 'Discriminator']
        if self.mode == 'enc':
            saver = tf.train.Saver(
                var_list=self.generator_vars + self.enc_params)
        else:
            saver = tf.train.Saver(var_list=self.generator_vars)
        self.load_generator = lambda ckpt_path=None: self.load(
            checkpoint_dir=ckpt_path, saver=saver)
        self._load_dataset()

    def _build_generator_discriminator(self):
        """Creates the generator and discriminator graph per dataset."""
        pass

    def _load_dataset(self):
        """Loads the dataset."""
        pass

    def _build(self):
        """Builds the computation graph."""

        assert (self.batch_size % self.rec_rr) == 0, 'Batch size ' \
                                                     'should be ' \
                                                     'divisable by ' \
                                                     'random restart'
        self.test_batch_size = self.batch_size

        # Defining batch_size in input placeholders is inevitable at least
        # for now, because the z vectors are Tensorflow variables.
        self.real_data_pl = tf.placeholder(
            tf.float32, shape=[self.batch_size] + self.image_dim,
        )
        self.real_data_test_pl = tf.placeholder(
            tf.float32, shape=[self.test_batch_size] + self.image_dim,
        )

        self.input_pl_transform()
        self._build_generator_discriminator()

        self.fake_data = self.generator_fn()

        self.disc_real = self.discriminator_fn(self.real_data)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            sc = tf.get_variable_scope()
            sc.reuse_variables()
            self.disc_fake = self.discriminator_fn(self.fake_data)

            self.generator_vars = slim.get_variables('Generator')
            self.discriminator_vars = slim.get_variables('Discriminator')

            self.fixed_noise = tf.constant(
                np.random.normal(size=(128, self.latent_dim)).astype(
                    'float32'))
            self.fixed_noise_samples = self.generator_fn(self.fixed_noise,
                                                         is_training=False)

    def _loss(self):
        """Builds the loss part of the graph.."""
        self.discriminator_cost = 0
        self.generator_cost = 0

        if self.mode == 'wgan':
            self.generator_cost = -tf.reduce_mean(self.disc_fake)
            self.discriminator_cost = tf.reduce_mean(
                self.disc_fake) - tf.reduce_mean(
                self.disc_real)

            self.gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.generator_cost, var_list=self.generator_vars)
            self.disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
            ).minimize(self.discriminator_cost,
                       var_list=self.discriminator_vars)

            clip_ops = []
            for var in tflib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(
                        var,
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            self.clip_disc_weights = tf.group(*clip_ops)

        elif self.mode == 'wgan-gp':
            self.generator_cost = -tf.reduce_mean(self.disc_fake)
            disc_cost = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(
                self.disc_real)

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1],
                minval=0.,
                maxval=1.
            )
            differences = self.fake_data - self.real_data
            interpolates = self.real_data + (alpha * differences)
            gradients = \
                tf.gradients(self.discriminator_fn(interpolates),
                             [interpolates])[0]
            slopes = tf.sqrt(
                tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.discriminator_cost = disc_cost + \
                                      self.gradient_penalty_lambda * \
                                      gradient_penalty

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.generator_cost, var_list=self.generator_vars)
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.discriminator_cost,
                       var_list=self.discriminator_vars)

            self.clip_disc_weights = None

        elif self.mode == 'dcgan':
            self.generator_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.disc_fake,
                    tf.ones_like(self.disc_fake)
                ))

            disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                self.disc_fake,
                tf.zeros_like(self.disc_fake)
            ))
            disc_cost += tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    self.disc_real,
                    tf.ones_like(self.disc_real)
                ))
            self.discriminator_cost = disc_cost / 2.

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5
            ).minimize(self.generator_cost, var_list=self.generator_vars)
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5
            ).minimize(disc_cost, var_list=self.discriminator_vars)

            self.clip_disc_weights = None

    def _generate_image(self, training_iter):
        """Generates a set of sample images from fixed noise and log them in
            the `debug` directory.

        Args:
            training_iter: The training iteration to include as part of the
                filename.
        """
        samples = self.sess.run(self.fixed_noise_samples)
        tflib.save_images.save_images(
            samples.reshape((128, 28, 28)),
            os.path.join(self.checkpoint_dir.replace('output', 'debug'),
                         'samples_{}.png'.format(training_iter))
        )

    def _inf_train_gen(self):
        """A generator function for input training data."""
        while True:
            for images, targets in self.train_data_gen():
                yield images

    def train(self, phase=None):
        """Trains the GAN model."""

        sess = self.sess
        self.initialize_uninitialized()

        gen = self._inf_train_gen()
        could_load = self.load(checkpoint_dir=self.checkpoint_dir,
                               prefixes=self.save_var_prefixes)
        if could_load:
            print('[*] Model loaded.')
        else:
            print('[#] No model found')

        cur_iter = self.sess.run(self.global_step)
        max_train_iters = self.train_iters
        step_inc = self.global_step_inc
        global_step = self.global_step
        ckpt_dir = self.checkpoint_dir

        for iteration in xrange(cur_iter, max_train_iters):
            start_time = time.time()

            if iteration > 0 and 'gan' in self.mode and phase is None:
                _ = sess.run(self.gen_train_op,
                             feed_dict={self.is_training: 1})

            if self.mode == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = self.critic_iters

            for i in xrange(disc_iters):
                _data = gen.next()
                _disc_cost, _ = sess.run(
                    [self.discriminator_cost, self.disc_train_op],
                    feed_dict={self.real_data_pl: _data,
                               self.is_training: 1}
                )
                if self.clip_disc_weights is not None:
                    _ = sess.run(self.clip_disc_weights)

            tflib.plot.plot('{}/train disc cost'.format(self.debug_dir),
                            _disc_cost)
            tflib.plot.plot('{}/time'.format(self.debug_dir),
                            time.time() - start_time)

            # Calculate dev loss and generate samples every 100 iters.
            if iteration % 100 == 5:
                dev_disc_costs = []
                dev_ctr = 0
                for images, _ in self.dev_gen():
                    dev_ctr += 1
                    if dev_ctr > 20:
                        break
                    _dev_disc_cost = sess.run(
                        self.discriminator_cost,
                        feed_dict={self.real_data_pl: images,
                                   self.is_training: 0}
                    )
                    dev_disc_costs.append(_dev_disc_cost)
                tflib.plot.plot('{}/dev disc cost'.format(self.debug_dir),
                                np.mean(dev_disc_costs))
                self.generate_image(iteration)

            # Write logs every 100 iters

            if (iteration < 5) or (iteration % 100 == 99):
                tflib.plot.flush()

            self.sess.run(step_inc)
            if iteration % 500 == 499:
                self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

            tflib.plot.tick()

        self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

        self.close_session()

    def reconstruct(
        self, images, batch_size=None, back_prop=True,
        reconstructor_id=0, z_init_val=None):
        """Creates the reconstruction op for Defense-GAN.

        Args:
            X: Input tensor

        Returns:
            The `tf.Tensor` of the reconstructed input.
        """

        # Batch size is needed because the latent codes are `tf.Variable`s and
        # need to be built into TF's static graph beforehand.

        batch_size = batch_size if batch_size else self.test_batch_size

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        # Repeat images self.rec_rr times to handle random restarts in
        # parallel.
        images_tiled_rr = tf.reshape(
            images, [x_shape[0], np.prod(x_shape[1:])])
        images_tiled_rr = tf.tile(images_tiled_rr, [1, self.rec_rr])
        images_tiled_rr = tf.reshape(
            images_tiled_rr, [x_shape[0] * self.rec_rr] + x_shape[1:])

        # Number of reconstruction iterations.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            rec_iter_const = tf.get_variable(
                'rec_iter_{}'.format(reconstructor_id),
                initializer=tf.constant(0),
                trainable=False, dtype=tf.int32,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            # The latent variables.
            z_hat = tf.get_variable(
                'z_hat_rec_{}'.format(reconstructor_id),
                shape=[batch_size * self.rec_rr, self.latent_dim],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1.0 / self.latent_dim)),
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

        # Learning rate for reconstruction.
        rec_lr_op_from_const = self.get_learning_rate(init_lr=self.rec_lr,
                                                      global_step=rec_iter_const,
                                                      decay_mult=0.1,
                                                      decay_iter=np.ceil(
                                                          self.rec_iters *
                                                          0.8).astype(
                                                          np.int32))

        # The optimizer.
        rec_online_optimizer = tf.train.MomentumOptimizer(
            learning_rate=rec_lr_op_from_const, momentum=0.7,
            name='rec_optimizer')



        init_z = tf.no_op()
        if z_init_val is not None:
            init_z = tf.assign(z_hat, z_init_val)

        z_hats_recs = self.generator_fn(z_hat, is_training=False)
        num_dim = len(z_hats_recs.get_shape())
        axes = range(1, num_dim)

        image_rec_loss = tf.reduce_mean(
            tf.square(z_hats_recs - images_tiled_rr),
            axis=axes)
        rec_loss = tf.reduce_sum(image_rec_loss)
        rec_online_optimizer.minimize(rec_loss, var_list=[z_hat])

        def rec_body(i, *args):
            z_hats_recs = self.generator_fn(z_hat, is_training=False)
            image_rec_loss = tf.reduce_mean(
                tf.square(z_hats_recs - images_tiled_rr),
                axis=axes)
            rec_loss = tf.reduce_sum(image_rec_loss)

            train_op = rec_online_optimizer.minimize(rec_loss,
                                                     var_list=[z_hat])

            return tf.tuple(
                [tf.add(i, 1), rec_loss, image_rec_loss, z_hats_recs],
                control_inputs=[train_op])

        rec_iter_condition = lambda i, *args: tf.less(i, self.rec_iters)
        for opt_var in rec_online_optimizer.variables():
            tf.add_to_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                opt_var,
            )

        with tf.control_dependencies([init_z]):
            online_rec_iter, online_rec_loss, online_image_rec_loss, \
            all_z_recs = tf.while_loop(
                rec_iter_condition,
                rec_body,
                [rec_iter_const, rec_loss, image_rec_loss, z_hats_recs]
                , parallel_iterations=1, back_prop=back_prop,
                swap_memory=False)
            final_recs = []
            for i in range(batch_size):
                ind = i * self.rec_rr + tf.argmin(
                    online_image_rec_loss[
                    i * self.rec_rr:(i + 1) * self.rec_rr
                    ],
                    axis=0)
                final_recs.append(all_z_recs[tf.cast(ind, tf.int32)])

            online_rec = tf.stack(final_recs)

            return tf.reshape(online_rec, x_shape)

    def reconstruct_dataset(self, ckpt_path=None, max_num=-1, max_num_load=-1):
        """Reconstructs the images of the config's dataset with the generator.
        """

        if not self.initialized:
            self.load_generator(ckpt_path=ckpt_path)

        splits = ['train', 'dev', 'test']

        rec = self.reconstruct(self.real_data_test)

        self.sess.run(tf.local_variables_initializer())
        rets = {}

        for split in splits:
            if max_num > 0:
                output_dir = os.path.join(self.checkpoint_dir,
                                          'recs_rr{:d}_lr{:.5f}_'
                                          'iters{:d}_num{:d}'.format(
                                              self.rec_rr, self.rec_lr,
                                              self.rec_iters, max_num),
                                          split)
            else:
                output_dir = os.path.join(self.checkpoint_dir,
                                          'recs_rr{:d}_lr{:.5f}_'
                                          'iters{:d}'.format(
                                              self.rec_rr, self.rec_lr,
                                              self.rec_iters), split)

            if self.debug:
                output_dir += '_debug'

            ensure_dir(output_dir)
            feats_path = os.path.join(output_dir, 'feats.pkl'.format(split))
            could_load = False
            try:
                if os.path.exists(feats_path) and not self.test_again:
                    with open(feats_path) as f:
                        all_recs = cPickle.load(f)
                        could_load = True
                        print('[#] Successfully loaded features.')
                else:
                    all_recs = []
            except Exception as e:
                all_recs = []
                print('[#] Exception loading features {}'.format(str(e)))

            gen_func = getattr(self, '{}_gen_test'.format(split))
            all_targets = []
            orig_imgs = []
            ctr = 0
            sti = time.time()

            # Pickle files per reconstructed image.
            pickle_out_dir = os.path.join(output_dir, 'pickles')
            ensure_dir(pickle_out_dir)
            single_feat_path_template = os.path.join(pickle_out_dir,
                                                     'rec_{:07d}_l{}.pkl')

            for images, targets in gen_func():
                batch_size = len(images)
                im_paths = [
                    single_feat_path_template.format(ctr * batch_size + i,
                                                     targets[i]) for i in
                    range(batch_size)]

                mn = max(max_num, max_num_load)

                if (mn > -1 and ctr * (len(images)) > mn) or (
                    self.debug and ctr > 2):
                    break

                batch_could_load = not self.test_again
                batch_rec_list = []

                for imp in im_paths: # Load per image cached files.
                    try:
                        with open(imp) as f:
                            loaded_rec = cPickle.load(f)
                            batch_rec_list.append(loaded_rec)
                            # print('[-] Loaded batch {}'.format(ctr))
                    except:
                        batch_could_load = False
                        break

                if batch_could_load and not could_load:
                    recs = np.stack(batch_rec_list)
                    all_recs.append(recs)

                if not (could_load or batch_could_load):
                    self.sess.run(tf.local_variables_initializer())
                    recs = self.sess.run(
                        rec, feed_dict={self.real_data_test_pl: images},
                    )
                    print('[#] t:{:.2f} batch: {:d} '.format(time.time() - sti,
                                                             ctr))
                    all_recs.append(recs)
                else:
                    print('[*] could load batch: {:d}'.format(ctr))

                if not batch_could_load and not could_load:
                    for i in range(len(recs)):
                        pkl_path = im_paths[i]
                        with open(pkl_path, 'w') as f:
                            cPickle.dump(recs[i], f,
                                         protocol=cPickle.HIGHEST_PROTOCOL)
                            #print('[*] Saved reconstruction for {}'.format(pkl_path))

                all_targets.append(targets)

                orig_transformed = self.sess.run(self.real_data_test,
                                                 feed_dict={
                                                     self.real_data_test_pl:
                                                         images})

                orig_imgs.append(orig_transformed)
                ctr += 1
            if not could_load:
                all_recs = np.concatenate(all_recs)
                all_recs = all_recs.reshape([-1] + self.image_dim)

            orig_imgs = np.concatenate(orig_imgs).reshape(
                [-1] + self.image_dim)
            all_targets = np.concatenate(all_targets)

            if self.debug:
                save_images_files(all_recs,
                                  output_dir=output_dir, labels=all_targets)
                save_images_files(
                    (orig_imgs + min(0, orig_imgs.min()) / (
                        orig_imgs.max() - min(0, orig_imgs.min()))),
                    output_dir=output_dir,
                    labels=all_targets, postfix='_orig')

            rets[split] = [all_recs, all_targets, orig_imgs]

        return rets

    def generate_image(self, iteration=None):
        """Generates a fixed noise for visualization of generation output.
        """
        pass

    def test_batch(self):
        """Tests the image batch generator."""
        output_dir = os.path.join(self.debug_dir, 'test_batch')
        ensure_dir(output_dir)

        img, target = self.train_data_gen().next()
        img = img.reshape([self.batch_size] + self.image_dim)
        save_images_files(img / 255.0, output_dir=output_dir,
                          labels=target)

    def save_ds(self):
        """Reconstructs the images of the config's dataset with the
        generator."""
        if self.dataset_name == 'cifar':
            splits = ['train', 'dev']
        else:
            splits = ['train', 'dev', 'test']
        for split in splits:
            output_dir = os.path.join('data', 'cache',
                                      '{}_pkl'.format(self.dataset_name),
                                      split)
            if self.debug:
                output_dir += '_debug'

            ensure_dir(output_dir)
            orig_imgs_pkl_path = os.path.join(output_dir,
                                              'feats.pkl'.format(split))

            if os.path.exists(orig_imgs_pkl_path) and not self.test_again:
                with open(orig_imgs_pkl_path) as f:
                    all_recs = cPickle.load(f)
                    could_load = True
                    print('[#] Dataset is already saved.')
                    return

            gen_func = getattr(self, '{}_gen_test'.format(split))
            all_targets = []
            orig_imgs = []
            ctr = 0
            for images, targets in gen_func():
                ctr += 1
                transformed_images = self.sess.run(self.real_data_test,
                                                   feed_dict={
                                                       self.real_data_test_pl:
                                                           images})
                orig_imgs.append(transformed_images)
                all_targets.append(targets)
            orig_imgs = np.concatenate(orig_imgs).reshape(
                [-1] + self.image_dim)
            all_targets = np.concatenate(all_targets)
            with open(orig_imgs_pkl_path, 'w') as f:
                cPickle.dump(orig_imgs, f, cPickle.HIGHEST_PROTOCOL)
                cPickle.dump(all_targets, f, cPickle.HIGHEST_PROTOCOL)


class MnistDefenseGAN(DefenseGANBase):
    def _build_generator_discriminator(self):
        self.discriminator_fn = lambda x: mnist_discriminator(
            x,
            use_bn=self.use_bn,
            net_dim=self.net_dim,
            is_training=self.is_training)

        self.generator_fn = lambda z=None, is_training=self.is_training: \
            mnist_generator(
                self.batch_size,
                use_bn=self.use_bn,
                net_dim=self.net_dim,
                is_training=is_training,
                latent_dim=self.latent_dim,
                output_dim=self.image_dim,
                noise=z)

    def _load_dataset(self):
        self.train_data_gen, self.dev_gen, _ = get_generators('mnist',
                                                              self.batch_size)
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = \
            get_generators(
                'mnist', self.test_batch_size,
                randomize=False)

    def generate_image(self, iteration):
        samples = self.sess.run(self.fixed_noise_samples)

        tflib.save_images.save_images(
            samples.reshape((len(samples), 28, 28)),
            os.path.join(self.checkpoint_dir.replace('output', 'debug'),
                         'samples_{}.png'.format(iteration))
        )

    def input_transform(self, X):
        return (tf.cast(X, tf.float32) / 255.)


class FmnistDefenseDefenseGAN(MnistDefenseGAN):
    def _load_dataset(self):
        self.train_data_gen, self.dev_gen, _ = get_generators('f-mnist',
                                                              self.batch_size)
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = \
            get_generators(
                'f-mnist', self.test_batch_size,
                randomize=False)

    def input_transform(self, X):
        return (tf.cast(X, tf.float32) / 255.)

    def generate_image(self, training_iter):
        samples = self.sess.run(self.fixed_noise_samples)

        tflib.save_images.save_images(
            samples.reshape((len(samples), 28, 28)),
            os.path.join(self.checkpoint_dir.replace('output', 'debug'),
                         'samples_{}.png'.format(training_iter))
        )
        if self.mode == 'enc':
            tflib.save_images.save_images(
                self.test_decoder_images.reshape(
                    (len(samples), 28, 28)) / 255.0,
                os.path.join(self.checkpoint_dir.replace('output', 'debug'),
                             'orig_{}.png'.format(training_iter))
            )


class CelebADefenseGAN(DefenseGANBase):
    def _build_generator_discriminator(self):
        self.discriminator_fn = lambda x: celeba_discriminator(
            x,
            use_bn=self.use_bn,
            net_dim=self.net_dim,
            is_training=self.is_training,
            stats_iter=self.global_step,
            data_format='NHWC')
        self.generator_fn = lambda z=None, is_training=self.is_training: \
            celeba_generator(
                self.batch_size,
                use_bn=self.use_bn,
                net_dim=self.net_dim,
                is_training=is_training,
                latent_dim=self.latent_dim,
                output_dim=self.image_dim,
                noise=z,
                stats_iter=self.global_step)

    def _load_dataset(self):
        self.train_data_gen, self.dev_gen, self.test_gen = get_generators(
            self.dataset_name, self.batch_size,
            attribute=self.attribute)
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = \
            get_generators(
                self.dataset_name,
                self.test_batch_size,
                randomize=False,
                attribute=self.attribute)
        self.test_decoder_images, _ = self.dev_gen().next()

    def generate_image(self, training_iter):
        samples = self.sess.run(self.fixed_noise_samples)
        debug_dir = self.checkpoint_dir.replace('output', 'debug')
        ensure_dir(debug_dir)
        tflib.save_images.save_images(
            (samples.reshape((len(samples), 64, 64, 3)) + 1) / (2.0),
            os.path.join(debug_dir, 'samples_{}.png'.format(training_iter))
        )

    def imsave_transform(self, imgs):
        imgs = (imgs + 1.0) / 2
        imgs[imgs < 0] = 0.0
        imgs[imgs > 1] = 1.0
        return imgs

    def input_transform(self, images):
        return 2 * ((tf.cast(images, tf.float32) / 255.) - .5)
