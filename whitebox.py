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

"""Testing white-box attacks Defense-GAN models. This module is based on MNIST
tutorial of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import _init_paths

import argparse
import cPickle
import logging
import os
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf

from blackbox import dataset_gan_dict, get_cached_gan_data
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train, model_eval
from models.gan import MnistDefenseGAN, FmnistDefenseDefenseGAN, CelebADefenseGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.network_builder import model_a, model_b, model_c, model_d, model_e, model_f

ds_gan = {
    'mnist': MnistDefenseGAN,
    'f-mnist': FmnistDefenseDefenseGAN,
    'celeba': CelebADefenseGAN,
}
orig_data_paths = {k: 'data/cache/{}_pkl'.format(k) for k in ds_gan.keys()}


def whitebox(gan, rec_data_path=None, batch_size=128, learning_rate=0.001,
             nb_epochs=10, eps=0.3, online_training=False,
             test_on_dev=True, attack_type='fgsm', defense_type='gan',
             num_tests=-1, num_train=-1):
    """Based on MNIST tutorial from cleverhans.
    
    Args:
         gan: A `GAN` model.
         rec_data_path: A string to the directory.
         batch_size: The size of the batch.
         learning_rate: The learning rate for training the target models.
         nb_epochs: Number of epochs for training the target model.
         eps: The epsilon of FGSM.
         online_training: Training Defense-GAN with online reconstruction. The
            faster but less accurate way is to reconstruct the dataset once and use
            it to train the target models with:
            `python train.py --cfg <path-to-model> --save_recs`
         attack_type: Type of the white-box attack. It can be `fgsm`,
            `rand+fgsm`, or `cw`.
         defense_type: String representing the type of attack. Can be `none`,
            `defense_gan`, or `adv_tr`.
    """
    
    FLAGS = tf.flags.FLAGS

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)

    if defense_type == 'defense_gan':
        assert gan is not None

    # Create TF session.
    if defense_type == 'defense_gan':
        sess = gan.sess
        if FLAGS.train_on_recs:
            assert rec_data_path is not None or online_training
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    train_images, train_labels, test_images, test_labels = \
        get_cached_gan_data(gan, test_on_dev)

    rec_test_images = test_images
    rec_test_labels = test_labels

    _, _, test_images, test_labels = \
        get_cached_gan_data(gan, test_on_dev, orig_data_flag=True)

    x_shape = [None] + list(train_images.shape[1:])
    images_pl = tf.placeholder(tf.float32, shape=[None] + list(train_images.shape[1:]))
    labels_pl = tf.placeholder(tf.float32, shape=[None] + [train_labels.shape[1]])

    if num_tests > 0:
        test_images = test_images[:num_tests]
        rec_test_images = rec_test_images[:num_tests]
        test_labels = test_labels[:num_tests]

    if num_train > 0:
        train_images = train_images[:num_train]
        train_labels = train_labels[:num_train]

    # GAN defense flag.
    models = {'A': model_a, 'B': model_b, 'C': model_c, 'D': model_d, 'E': model_e, 'F': model_f}
    model = models[FLAGS.model](input_shape=x_shape, nb_classes=train_labels.shape[1])

    preds = model.get_probs(images_pl)
    report = AccuracyReport()

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test
        # examples.
        eval_params = {'batch_size': batch_size}
        acc = model_eval(
            sess, images_pl, labels_pl, preds, rec_test_images,
            rec_test_labels, args=eval_params,
            feed={K.learning_phase(): 0})
        report.clean_train_clean_eval = acc
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }

    rng = np.random.RandomState([11, 24, 1990])
    tf.set_random_seed(11241990)

    preds_adv = None
    if FLAGS.defense_type == 'adv_tr':
        attack_params = {'eps': FLAGS.fgsm_eps_tr,
                         'clip_min': 0.,
                         'clip_max': 1.}
        if gan:
            if gan.dataset_name == 'celeba':
                attack_params['clip_min'] = -1.0

        attack_obj = FastGradientMethod(model, sess=sess)
        adv_x_tr = attack_obj.generate(images_pl, **attack_params)
        adv_x_tr = tf.stop_gradient(adv_x_tr)
        preds_adv = model(adv_x_tr)

    model_train(sess, images_pl, labels_pl, preds, train_images, train_labels,
                args=train_params, rng=rng, predictions_adv=preds_adv,
                init_all=False, feed={K.learning_phase(): 1},
                evaluate=evaluate)

    # Calculate training error.
    eval_params = {'batch_size': batch_size}
    acc = model_eval(
        sess, images_pl, labels_pl, preds, train_images, train_labels,
        args=eval_params, feed={K.learning_phase(): 0},
    )
    print('[#] Accuracy on clean examples {}'.format(acc))
    if attack_type is None:
        return acc, 0, None

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph.

    if FLAGS.defense_type == 'defense_gan':
        z_init_val = None

        if FLAGS.same_init:
            z_init_val = tf.constant(
                np.random.randn(batch_size * gan.rec_rr, gan.latent_dim).astype(np.float32))

        model.add_rec_model(gan, z_init_val, batch_size)

    min_val = 0.0
    if gan:
        if gan.dataset_name == 'celeba':
            min_val = -1.0

    if 'rand' in FLAGS.attack_type:
        test_images = np.clip(
            test_images + args.alpha * np.sign(np.random.randn(*test_images.shape)),
            min_val, 1.0)
        eps -= args.alpha

    if 'fgsm' in FLAGS.attack_type:
        attack_params = {'eps': eps, 'ord': np.inf, 'clip_min': min_val, 'clip_max': 1.}
        attack_obj = FastGradientMethod(model, sess=sess)
    elif FLAGS.attack_type == 'cw':
        attack_obj = CarliniWagnerL2(model, back='tf', sess=sess)
        attack_iterations = 100
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 10.0,
                         'batch_size': batch_size,
                         'initial_const': 100,
                         'feed': {K.learning_phase(): 0}}
    adv_x = attack_obj.generate(images_pl, **attack_params)

    eval_par = {'batch_size': batch_size}
    if FLAGS.defense_type == 'defense_gan':
        preds_adv = model.get_probs(adv_x)

        num_dims = len(images_pl.get_shape())
        avg_inds = list(range(1, num_dims))
        diff_op = tf.reduce_mean(tf.square(adv_x - images_pl), axis=avg_inds)
        acc_adv, roc_info = model_eval_gan(
            sess, images_pl, labels_pl, preds_adv, None,
            test_images=test_images, test_labels=test_labels, args=eval_par,
            feed={K.learning_phase(): 0}, diff_op=diff_op,
        )
        print('Test accuracy on adversarial examples: %0.4f\n' % acc_adv)
        return acc_adv, 0, roc_info
    else:
        preds_adv = model(adv_x)
        acc_adv = model_eval(sess, images_pl, labels_pl, preds_adv, test_images, test_labels,
                             args=eval_par,
                             feed={K.learning_phase(): 0})
        print('Test accuracy on adversarial examples: %0.4f\n' % acc_adv)

        return acc_adv, 0, None


import re


def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS
    GAN = dataset_gan_dict[FLAGS.dataset_name]

    gan = GAN(cfg=cfg, test_mode=True)
    gan.load_generator()
    # Setting test time reconstruction hyper parameters.
    [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
    if FLAGS.defense_type.lower() != 'none':
        if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':

            # Extract hyperparameters from reconstruction path.
            if FLAGS.rec_path:
                train_param_re = re.compile('recs_rr(.*)_lr(.*)_iters(.*)')
                [tr_rr, tr_lr, tr_iters] = \
                    train_param_re.findall(FLAGS.rec_path)[0]
                gan.rec_rr = int(tr_rr)
                gan.rec_lr = float(tr_lr)
                gan.rec_iters = int(tr_iters)
        elif FLAGS.defense_type == 'defense_gan':
            assert FLAGS.online_training or not FLAGS.train_on_recs

    if FLAGS.override:
        gan.rec_rr = int(tr_rr)
        gan.rec_lr = float(tr_lr)
        gan.rec_iters = int(tr_iters)

    # Setting the results directory.
    results_dir, result_file_name = _get_results_dir_filename(gan)

    # Result file name. The counter ensures we are not overwriting the
    # results.
    counter = 0
    temp_fp = str(counter) + '_' + result_file_name
    results_dir = os.path.join(results_dir, FLAGS.results_dir)
    temp_final_fp = os.path.join(results_dir, temp_fp)
    while os.path.exists(temp_final_fp):
        counter += 1
        temp_fp = str(counter) + '_' + result_file_name
        temp_final_fp = os.path.join(results_dir, temp_fp)
    result_file_name = temp_fp
    sub_result_path = os.path.join(results_dir, result_file_name)

    accuracies = whitebox(
        gan, rec_data_path=FLAGS.rec_path,
        batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        nb_epochs=FLAGS.nb_epochs,
        eps=FLAGS.fgsm_eps,
        online_training=FLAGS.online_training,
        defense_type=FLAGS.defense_type,
        num_tests=FLAGS.num_tests,
        attack_type=FLAGS.attack_type,
        num_train=FLAGS.num_train,
    )

    ensure_dir(results_dir)

    with open(sub_result_path, 'a') as f:
        f.writelines([str(accuracies[i]) + ' ' for i in range(2)])
        f.write('\n')
        print('[*] saved accuracy in {}'.format(sub_result_path))

    if accuracies[2]:  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc.pkl')
        with open(pkl_result_path, 'w') as f:
            cPickle.dump(accuracies[2], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info in {}'.format(pkl_result_path))


def _get_results_dir_filename(gan):
    FLAGS = tf.flags.FLAGS

    results_dir = os.path.join('results', 'whitebox_{}_{}'.format(
        FLAGS.defense_type, FLAGS.dataset_name))

    if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':
        results_dir = gan.checkpoint_dir.replace('output', 'results')
        result_file_name = \
            'Iter={}_RR={:d}_LR={:.4f}_defense=gan'.format(
                gan.rec_rr,
                gan.rec_lr,
                gan.rec_iters,
                FLAGS.attack_type,
            )

        if not FLAGS.train_on_recs:
            result_file_name = 'orig_' + result_file_name
    elif FLAGS.defense_type == 'adv_tr':
        result_file_name = 'advTrEps={:.2f}'.format(FLAGS.fgsm_eps_tr)
    else:
        result_file_name = 'nodefense_'
    if FLAGS.num_tests > -1:
        result_file_name = 'numtest={}_'.format(
            FLAGS.num_tests) + result_file_name

    if FLAGS.num_train > -1:
        result_file_name = 'numtrain={}_'.format(
            FLAGS.num_train) + result_file_name

    result_file_name = 'model={}_'.format(FLAGS.model) + result_file_name
    result_file_name += 'attack={}.txt'.format(FLAGS.attack_type)
    return results_dir, result_file_name


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="RAND+FGSM random perturbation scale")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python whitebox.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train model.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697.')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_string('rec_path', None, 'Path to reconstructions.')
    flags.DEFINE_integer('num_tests', -1, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the classifier.')
    flags.DEFINE_boolean("online_training", False,
                         "Train the base classifier on reconstructions.")
    flags.DEFINE_string("defense_type", "none", "Type of defense [none|defense_gan|adv_tr]")
    flags.DEFINE_string("attack_type", "none", "Type of attack [fgsm|cw|rand_fgsm]")
    flags.DEFINE_string("results_dir", None, "The final subdirectory of the results.")
    flags.DEFINE_boolean("same_init", False, "Same initialization for z_hats.")
    flags.DEFINE_string("model", "F", "The classifier model.")
    flags.DEFINE_string("debug_dir", "temp", "The debug directory.")
    flags.DEFINE_integer("num_train", -1, 'Number of training data to load.')
    flags.DEFINE_boolean("debug", False, "True for saving reconstructions [False]")
    flags.DEFINE_boolean("override", False, "Overriding the config values of reconstruction "
                                            "hyperparameters. It has to be true if either "
                                            "`--rec_rr`, `--rec_lr`, or `--rec_iters` is passed "
                                            "from command line.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the classifier on the reconstructed samples "
                         "using Defense-GAN.")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
