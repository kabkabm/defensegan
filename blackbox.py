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

"""Testing blackbox Defense-GAN models. This module is based on MNIST tutorial
of cleverhans."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle
import logging
import os
import re
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf
from six.moves import xrange
from tensorflow.python.platform import flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation
from cleverhans.utils import set_log_level, to_categorical
from cleverhans.utils_tf import model_train, model_eval, batch_eval
from datasets.celeba import CelebA
from datasets.dataset import PickleLazyDataset
from models.gan import MnistDefenseGAN, FmnistDefenseDefenseGAN, \
    CelebADefenseGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.misc import ensure_dir
from utils.network_builder import model_a, model_b, model_c, model_d, \
    model_e, model_f, model_z, model_q
from utils.visualize import save_images_files

FLAGS = flags.FLAGS
dataset_gan_dict = {
    'mnist': MnistDefenseGAN,
    'f-mnist': FmnistDefenseDefenseGAN,
    'celeba': CelebADefenseGAN,
}

# orig_ refers to original images and not reconstructed ones.
# To prepare these cache files run "python main.py --save_ds".
orig_data_path = {k: 'data/cache/{}_pkl'.format(k) for k in
                  dataset_gan_dict.keys()}


def prep_bbox(sess, images, labels, images_train, labels_train, images_test,
              labels_test, nb_epochs, batch_size, learning_rate, rng, gan=None,
              adv_training=False, cnn_arch=None):
    """Defines and trains a model that simulates the "remote"
    black-box oracle described in https://arxiv.org/abs/1602.02697.
    
    Args:
        sess: the TF session
        images: the input placeholder
        labels: the ouput placeholder
        images_train: the training data for the oracle
        labels_train: the training labels for the oracle
        images_test: the testing data for the oracle
        labels_test: the testing labels for the oracle
        nb_epochs: number of epochs to train model
        batch_size: size of training batches
        learning_rate: learning rate for training
        rng: numpy.random.RandomState
    
    Returns:
        model: The blackbox model function.
        predictions: The predictions tensor.
        accuracy: Accuracy of the model.
    """

    # Define TF model graph (for the black-box model).
    model = cnn_arch
    if gan:
        x_rec = tf.stop_gradient(
            gan.reconstruct(images, batch_size=batch_size))
        predictions = model(x_rec)
    else:
        predictions = model(images)
    print("Defined TensorFlow model graph.")

    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    preds_adv = None

    if adv_training:

        fgsm_par = {'eps': FLAGS.fgsm_eps_tr, 'ord': np.inf, 'clip_min': 0.,
                    'clip_max': 1.}
        if gan:
            if any([xx in gan.dataset_name for xx in ['celeba']]):
                fgsm_par['clip_min'] = -1.0
        fgsm_params = fgsm_par

        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(images, **fgsm_params)
        adv_x = tf.stop_gradient(adv_x)
        preds_adv = model(adv_x)

    model_train(
        sess, images, labels, predictions, images_train, labels_train,
        args=train_params, rng=rng, predictions_adv=preds_adv,
        init_all=False, feed={K.learning_phase(): 1}
    )

    # Print out the accuracy on legitimate test data.
    eval_params = {'batch_size': batch_size}

    accuracy = model_eval(
        sess, images, labels, predictions, images_test,
        labels_test, args=eval_params, feed={K.learning_phase(): 0},
    )

    print(
        'Test accuracy of black-box on legitimate test examples: ' +
        str(accuracy)
    )

    return model, predictions, accuracy


def train_sub(sess, x, y, bbox_preds, X_sub, Y_sub, nb_classes,
              nb_epochs_s, batch_size, learning_rate, data_aug, lmbda,
              rng, substitute_model=None):
    """This function trains the substitute model as described in
        arxiv.org/abs/1602.02697

    Args:
        sess: TF session
        x: input TF placeholder
        y: output TF placeholder
        bbox_preds: output of black-box model predictions
        X_sub: initial substitute training data
        Y_sub: initial substitute training labels
        nb_classes: number of output classes
        nb_epochs_s: number of epochs to train substitute model
        batch_size: size of training batches
        learning_rate: learning rate for training
        data_aug: number of times substitute training data is augmented
        lmbda: lambda from arxiv.org/abs/1602.02697
        rng: numpy.random.RandomState instance
    
    Returns:
        model_sub: The substitute model function.
        preds_sub: The substitute prediction tensor.
    """
    # Define TF model graph (for the black-box model).
    model_sub = substitute_model
    preds_sub = model_sub(x)
    print("Defined TensorFlow model graph for the substitute.")

    # Define the Jacobian symbolically using TensorFlow.
    grads = jacobian_graph(preds_sub, x, nb_classes)

    # Train the substitute and augment dataset alternatively.
    for rho in xrange(data_aug):
        print("Substitute training epoch #" + str(rho))
        train_params = {
            'nb_epochs': nb_epochs_s,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        model_train(sess, x, y, preds_sub, X_sub, to_categorical(Y_sub),
                    init_all=False, args=train_params,
                    rng=rng, feed={K.learning_phase(): 1})

        # If we are not at last substitute training iteration, augment dataset.
        if rho < data_aug - 1:

            print("Augmenting substitute training data.")
            # Perform the Jacobian augmentation.
            X_sub = jacobian_augmentation(sess, x, X_sub, Y_sub, grads, lmbda,
                                          feed={K.learning_phase(): 0})

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box.
            Y_sub = np.hstack([Y_sub, Y_sub])
            X_sub_prev = X_sub[int(len(X_sub) / 2):]
            eval_params = {'batch_size': batch_size}

            # To initialize the local variables of Defense-GAN.
            sess.run(tf.local_variables_initializer())

            bbox_val = batch_eval(sess, [x], [bbox_preds], [X_sub_prev],
                                  args=eval_params,
                                  feed={K.learning_phase(): 0})[0]
            # Note here that we take the argmax because the adversary
            # only has access to the label (not the probabilities) output
            # by the black-box model.
            Y_sub[int(len(X_sub) / 2):] = np.argmax(bbox_val, axis=1)

    return model_sub, preds_sub


def convert_to_onehot(ys):
    """Converts the labels to one-hot vectors."""
    max_y = int(np.max(ys))
    y_one_hat = np.zeros([len(ys), max_y + 1], np.float32)
    for (i, y) in enumerate(ys):
        y_one_hat[i, int(y)] = 1.0
    return y_one_hat


def get_celeba(data_path, test_on_dev=True, orig_data=False):
    """Generates the CelebA dataset from Pickle files.

    Args:
        data_path: The path to where pickles are saved.
            <model-path>/<split>/pickles/
        test_on_dev: Test on the development set.
        orig_data: Original data flag. `True` for returning the original
            dataset.

    Returns:
        images: Images of the dataset.
        labels: Labels of the loaded images.
    """
    dev_name = 'dev'
    if not test_on_dev:
        dev_name = 'test'
    ds = CelebA(attribute=FLAGS.attribute)
    ds.load()
    ds_test = CelebA(attribute=FLAGS.attribute)
    ds_test.load(split=dev_name, transform_type=1)
    train_labels = ds.labels
    test_labels = ds_test.labels

    def get_pickeldb(split):
        train_data_path = os.path.join(data_path, split, 'pickles')
        assert os.path.exists(train_data_path)
        pkl_files = os.listdir(train_data_path)
        pkl_labels = np.array(
            [int(re.findall('.*_l(\d+).pkl', pf)[0]) for pf in pkl_files],
            np.int32)
        pkl_paths = [os.path.join(train_data_path, pf) for pf in
                     sorted(pkl_files)]
        pkl_ds = PickleLazyDataset(pkl_paths, [64, 64, 3])
        return pkl_ds, pkl_labels

    if orig_data:
        train_images = ds.images
        test_images = ds_test.images
    else:
        train_images, train_labels = get_pickeldb('train')
        test_images, test_labels = get_pickeldb(dev_name)

    return train_images, convert_to_onehot(train_labels), test_images, \
           convert_to_onehot(test_labels)


def get_train_test(data_path, test_on_dev=True, model=None,
                   orig_data=False, max_num=-1):
    """Loads the datasets.

    Args:
        data_path: The path that contains train,dev,[test] directories
        test_on_dev: Test on the development set
        model: An instance of `GAN`.
        orig_data: `True` for loading original data, `False` to load the
            reconstructed images.
    
    Returns:
        train_images: Training images.
        train_labels: Training labels.
        test_images: Testing images.
        test_labels: Testing labels.
    """

    data_dict = None
    if model and not orig_data:
        data_dict = model.reconstruct_dataset(max_num_load=max_num)

    def get_images_labels_from_pickle(data_path, split):
        data_path = os.path.join(data_path, split, 'feats.pkl')
        could_load = False
        try:
            if os.path.exists(data_path):
                with open(data_path) as f:
                    train_images_gan = cPickle.load(f)
                    train_labels_gan = cPickle.load(f)
                could_load = True
            else:
                print(
                    '[!] Run python train.py --cfg <path-to-cfg> --save_ds '
                    'to prepare the dataset cache files.'
                )
                exit(1)

        except Exception as e:
            print(
                '[!] Found feats.pkl but could not load it because {}'.format(
                    str(e)))

        if not could_load and not data_dict is None:
            train_images_gan, train_labels_gan, train_images_orig = data_dict[
                split]
            if orig_data:
                train_images_gan = train_images_orig

        return train_images_gan, convert_to_onehot(train_labels_gan)

    train_images, train_lables = \
        get_images_labels_from_pickle(data_path, 'train')
    test_split = 'test' if test_on_dev else 'dev'
    test_images, test_labels = \
        get_images_labels_from_pickle(data_path, test_split)

    return train_images, train_lables, test_images, test_labels


def get_cached_gan_data(gan, test_on_dev, orig_data_flag=None):
    """Fetches the dataset of a GAN model.
    
    Args:
        gan: The GAN model.
        test_on_dev: `True` for loading the dev set instead of the test set.
        orig_data_flag: `True` for loading the original images not the 
            reconstructions.

    Returns:
        train_images: Training images.
        train_labels: Training labels.
        test_images: Testing images.
        test_labels: Testing labels.
    """
    FLAGS = flags.FLAGS
    if orig_data_flag is None:
        if not FLAGS.train_on_recs or FLAGS.defense_type != 'defense_gan':
            orig_data_flag = True
        else:
            orig_data_flag = False

    if 'celeba' in gan.dataset_name:
        train_images, train_labels, test_images, test_labels = get_celeba(
            FLAGS.rec_path,
            orig_data=orig_data_flag,
        )
        if FLAGS.num_train > 0:
            train_images = train_images[:FLAGS.num_train]
            train_labels = train_labels[:FLAGS.num_train]
    else:
        train_images, train_labels, test_images, test_labels = \
            get_train_test(
                orig_data_path[gan.dataset_name], test_on_dev=test_on_dev,
                model=gan, orig_data=orig_data_flag, max_num=FLAGS.num_train)
    return train_images, train_labels, test_images, test_labels


def blackbox(gan, rec_data_path=None, batch_size=128,
             learning_rate=0.001, nb_epochs=10, holdout=150, data_aug=6,
             nb_epochs_s=10, lmbda=0.1, online_training=False,
             train_on_recs=False, test_on_dev=True,
             defense_type='none'):
    """MNIST tutorial for the black-box attack from arxiv.org/abs/1602.02697
    
    Args:
        train_start: index of first training set example
        train_end: index of last training set example
        test_start: index of first test set example
        test_end: index of last test set example
        defense_type: Type of defense against blackbox attacks
    
    Returns:
        a dictionary with:
             * black-box model accuracy on test set
             * substitute model accuracy on test set
             * black-box model accuracy on adversarial examples transferred
               from the substitute model
    """
    FLAGS = flags.FLAGS

    # Set logging level to see debug information.
    set_log_level(logging.WARNING)

    # Dictionary used to keep track and return key accuracies.
    accuracies = {}

    # Create TF session.
    adv_training = False
    if defense_type:
        if defense_type == 'defense_gan' and gan:
            sess = gan.sess
            gan_defense_flag = True
        else:
            gan_defense_flag = False
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
        if 'adv_tr' in defense_type:
            adv_training = True
    else:
        gan_defense_flag = False
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    train_images, train_labels, test_images, test_labels = \
        get_cached_gan_data(gan, test_on_dev, orig_data_flag=True)

    x_shape, classes = list(train_images.shape[1:]), train_labels.shape[1]
    nb_classes = classes

    type_to_models = {
        'A': model_a, 'B': model_b, 'C': model_c, 'D': model_d, 'E': model_e,
        'F': model_f, 'Q': model_q, 'Z': model_z
    }

    bb_model = type_to_models[FLAGS.bb_model](
        input_shape=[None] + x_shape, nb_classes=train_labels.shape[1],
    )
    sub_model = type_to_models[FLAGS.sub_model](
        input_shape=[None] + x_shape, nb_classes=train_labels.shape[1],
    )

    if FLAGS.debug:
        train_images = train_images[:20 * batch_size]
        train_labels = train_labels[:20 * batch_size]
        debug_dir = os.path.join('debug', 'blackbox', FLAGS.debug_dir)
        ensure_dir(debug_dir)
        x_debug_test = test_images[:batch_size]

    # Initialize substitute training set reserved for adversary
    images_sub = test_images[:holdout]
    labels_sub = np.argmax(test_labels[:holdout], axis=1)

    # Redefine test set as remaining samples unavailable to adversaries
    if FLAGS.num_tests > 0:
        test_images = test_images[:FLAGS.num_tests]
        test_labels = test_labels[:FLAGS.num_tests]

    test_images = test_images[holdout:]
    test_labels = test_labels[holdout:]

    # Define input and output TF placeholders

    if FLAGS.image_dim[0] == 3:
        FLAGS.image_dim = [FLAGS.image_dim[1], FLAGS.image_dim[2],
                           FLAGS.image_dim[0]]

    images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
    labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

    rng = np.random.RandomState([11, 24, 1990])
    tf.set_random_seed(11241990)

    train_images_bb, train_labels_bb, test_images_bb, test_labels_bb = \
        train_images, train_labels, test_images, \
        test_labels

    cur_gan = None

    if defense_type:
        if 'gan' in defense_type:
            # Load cached dataset reconstructions.
            if online_training and not train_on_recs:
                cur_gan = gan
            elif not online_training and rec_data_path:
                train_images_bb, train_labels_bb, test_images_bb, \
                test_labels_bb = get_cached_gan_data(
                    gan, test_on_dev, orig_data_flag=False)
            else:
                assert not train_on_recs

        if FLAGS.debug:
            train_images_bb = train_images_bb[:20 * batch_size]
            train_labels_bb = train_labels_bb[:20 * batch_size]

        # Prepare the black_box model.
        prep_bbox_out = prep_bbox(
            sess, images_tensor, labels_tensor, train_images_bb,
            train_labels_bb, test_images_bb, test_labels_bb, nb_epochs,
            batch_size, learning_rate, rng=rng, gan=cur_gan,
            adv_training=adv_training,
            cnn_arch=bb_model)
    else:
        prep_bbox_out = prep_bbox(sess, images_tensor, labels_tensor,
                                  train_images_bb, train_labels_bb,
                                  test_images_bb, test_labels_bb,
                                  nb_epochs, batch_size, learning_rate,
                                  rng=rng, gan=cur_gan,
                                  adv_training=adv_training,
                                  cnn_arch=bb_model)

    model, bbox_preds, accuracies['bbox'] = prep_bbox_out

    # Train substitute using method from https://arxiv.org/abs/1602.02697
    print("Training the substitute model.")
    reconstructed_tensors = tf.stop_gradient(
        gan.reconstruct(images_tensor, batch_size=batch_size,
                        reconstructor_id=1))
    model_sub, preds_sub = train_sub(
        sess, images_tensor, labels_tensor,
        model(reconstructed_tensors), images_sub,
        labels_sub,
        nb_classes, nb_epochs_s, batch_size,
        learning_rate, data_aug, lmbda, rng=rng,
        substitute_model=sub_model,
    )

    accuracies['sub'] = 0
    # Initialize the Fast Gradient Sign Method (FGSM) attack object.
    fgsm_par = {
        'eps': FLAGS.fgsm_eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.
    }
    if gan:
        if gan.dataset_name == 'celeba':
            fgsm_par['clip_min'] = -1.0

    fgsm = FastGradientMethod(model_sub, sess=sess)

    # Craft adversarial examples using the substitute.
    eval_params = {'batch_size': batch_size}
    x_adv_sub = fgsm.generate(images_tensor, **fgsm_par)

    if FLAGS.debug and gan is not None:  # To see some qualitative results.
        reconstructed_tensors = gan.reconstruct(x_adv_sub, batch_size=batch_size,
                                                reconstructor_id=2)

        x_rec_orig = gan.reconstruct(images_tensor, batch_size=batch_size,
                                     reconstructor_id=3)
        x_adv_sub_val = sess.run(x_adv_sub,
                                 feed_dict={images_tensor: x_debug_test,
                                            K.learning_phase(): 0})
        sess.run(tf.local_variables_initializer())
        x_rec_debug_val, x_rec_orig_val = sess.run(
            [reconstructed_tensors, x_rec_orig],
            feed_dict={
                images_tensor: x_debug_test,
                K.learning_phase(): 0})

        save_images_files(x_adv_sub_val, output_dir=debug_dir,
                          postfix='adv')

        postfix = 'gen_rec'
        save_images_files(x_rec_debug_val, output_dir=debug_dir,
                          postfix=postfix)
        save_images_files(x_debug_test, output_dir=debug_dir,
                          postfix='orig')
        save_images_files(x_rec_orig_val, output_dir=debug_dir,
                          postfix='orig_rec')
        return

    if gan_defense_flag:
        reconstructed_tensors = gan.reconstruct(
            x_adv_sub, batch_size=batch_size, reconstructor_id=4,
        )

        num_dims = len(images_tensor.get_shape())
        avg_inds = list(range(1, num_dims))
        diff_op = tf.reduce_mean(tf.square(x_adv_sub - reconstructed_tensors),
                                 axis=avg_inds)

        outs = model_eval_gan(sess, images_tensor, labels_tensor,
                              predictions=model(reconstructed_tensors),
                              test_images=test_images, test_labels=test_labels,
                              args=eval_params, diff_op=diff_op,
                              feed={K.learning_phase(): 0})

        accuracies['bbox_on_sub_adv_ex'] = outs[0]
        accuracies['roc_info'] = outs[1]
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute: ' + str(outs[0]))
    else:
        accuracy = model_eval(sess, images_tensor, labels_tensor,
                              model(x_adv_sub), test_images,
                              test_labels,
                              args=eval_params, feed={K.learning_phase(): 0})
        print('Test accuracy of oracle on adversarial examples generated '
              'using the substitute: ' + str(accuracy))
        accuracies['bbox_on_sub_adv_ex'] = accuracy

    return accuracies


def _get_results_dir_filename(gan):
    result_file_name = 'sub={:d}_eps={:.2f}.txt'.format(FLAGS.data_aug,
                                                        FLAGS.fgsm_eps)

    results_dir = os.path.join('results', '{}_{}'.format(
        FLAGS.defense_type, FLAGS.dataset_name))

    if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':
        results_dir = gan.checkpoint_dir.replace('output', 'results')
        result_file_name = \
            'teRR={:d}_teLR={:.4f}_teIter={:d}_sub={:d}_eps={:.2f}.txt'.format(
                gan.rec_rr,
                gan.rec_lr,
                gan.rec_iters,
                FLAGS.data_aug,
                FLAGS.fgsm_eps)

        if not FLAGS.train_on_recs:
            result_file_name = 'orig_' + result_file_name
    elif FLAGS.defense_type == 'adv_tr':
        result_file_name = 'sub={:d}_trEps={:.2f}_eps={:.2f}.txt'.format(
            FLAGS.data_aug, FLAGS.fgsm_eps_tr,
            FLAGS.fgsm_eps)
    if FLAGS.num_tests > -1:
        result_file_name = 'numtest={}_'.format(
            FLAGS.num_tests) + result_file_name

    if FLAGS.num_train > -1:
        result_file_name = 'numtrain={}_'.format(
            FLAGS.num_train) + result_file_name

    result_file_name = 'bbModel={}_subModel={}_'.format(FLAGS.bb_model,
                                                        FLAGS.sub_model) \
                       + result_file_name
    return results_dir, result_file_name


def main(cfg, argv=None):
    FLAGS = tf.app.flags.FLAGS
    GAN = dataset_gan_dict[FLAGS.dataset_name]

    gan = GAN(cfg=cfg, test_mode=True)
    gan.load_generator()
    # Setting test time reconstruction hyper parameters.
    [tr_rr, tr_lr, tr_iters] = [FLAGS.rec_rr, FLAGS.rec_lr, FLAGS.rec_iters]
    if FLAGS.defense_type.lower() != 'none':
        if FLAGS.rec_path and FLAGS.defense_type == 'defense_gan':

            # extract hyper parameters from reconstruction path.
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

    # Setting the reuslts directory
    results_dir, result_file_name = _get_results_dir_filename(gan)

    # Result file name. The counter makes sure we are not overwriting the
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

    accuracies = blackbox(gan, rec_data_path=FLAGS.rec_path,
                          batch_size=FLAGS.batch_size,
                          learning_rate=FLAGS.learning_rate,
                          nb_epochs=FLAGS.nb_epochs, holdout=FLAGS.holdout,
                          data_aug=FLAGS.data_aug,
                          nb_epochs_s=FLAGS.nb_epochs_s,
                          lmbda=FLAGS.lmbda,
                          online_training=FLAGS.online_training,
                          train_on_recs=FLAGS.train_on_recs,
                          defense_type=FLAGS.defense_type)

    ensure_dir(results_dir)

    with open(sub_result_path, 'a') as f:
        f.writelines([str(accuracies[x]) + ' ' for x in
                      ['bbox', 'sub', 'bbox_on_sub_adv_ex']])
        f.write('\n')
        print('[*] saved accuracy in {}'.format(sub_result_path))

    if 'roc_info' in accuracies.keys():  # For attack detection.
        pkl_result_path = sub_result_path.replace('.txt', '_roc.pkl')
        with open(pkl_result_path, 'w') as f:
            cPickle.dump(accuracies['roc_info'], f, cPickle.HIGHEST_PROTOCOL)
            print('[*] saved roc_info in {}'.format(sub_result_path))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # Note: The load_config() call will convert all the parameters that are defined in
    # experiments/config files into FLAGS.param_name and can be passed in from command line.
    # arguments : python blackbox.py --cfg <config_path> --<param_name> <param_value>
    cfg = load_config(args.cfg)
    flags = tf.app.flags

    flags.DEFINE_integer('nb_classes', 10, 'Number of classes.')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training '
                                               'the black-box model.')
    flags.DEFINE_integer('nb_epochs', 10, 'Number of epochs to train the '
                                          'blackbox model.')
    flags.DEFINE_integer('holdout', 150, 'Test set holdout for adversary.')
    flags.DEFINE_integer('data_aug', 6, 'Number of substitute data augmentations.')
    flags.DEFINE_integer('nb_epochs_s', 10, 'Training epochs for substitute.')
    flags.DEFINE_float('lmbda', 0.1, 'Lambda from arxiv.org/abs/1602.02697')
    flags.DEFINE_float('fgsm_eps', 0.3, 'FGSM epsilon.')
    flags.DEFINE_float('fgsm_eps_tr', 0.15, 'FGSM epsilon for adversarial '
                                            'training.')
    flags.DEFINE_string('rec_path', None, 'Path to Defense-GAN '
                                          'reconstructions.')
    flags.DEFINE_integer('num_tests', 2000, 'Number of test samples.')
    flags.DEFINE_integer('random_test_iter', -1,
                         'Number of random sampling for testing the '
                         'classifier.')
    flags.DEFINE_boolean("online_training", False,
                         'Train the base classifier based on online '
                         'reconstructions from Defense-GAN, as opposed to '
                         'using the cached reconstructions.')
    flags.DEFINE_string("defense_type", "none", "Type of defense "
                                                "[defense_gan|adv_tr|none]")
    flags.DEFINE_string("results_dir", None, "The path to results.")
    flags.DEFINE_boolean("train_on_recs", False,
                         "Train the black-box model on Defense-GAN "
                         "reconstructions.")
    flags.DEFINE_integer('num_train', -1, 'Number of training samples for '
                                          'the black-box model.')
    flags.DEFINE_string("bb_model", 'F',
                        "The architecture of the classifier model.")
    flags.DEFINE_string("sub_model", 'E', "The architecture of the "
                                          "substitute model.")
    flags.DEFINE_string("debug_dir", None, "Directory for debug outputs.")
    flags.DEFINE_boolean("debug", None, "Directory for debug outputs.")
    flags.DEFINE_boolean("override", None, "Overrides the test hyperparams.")

    main_cfg = lambda x: main(cfg, x)
    tf.app.run(main=main_cfg)
