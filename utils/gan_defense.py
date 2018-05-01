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

"""Defense-GAN model evaluation function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import tensorflow as tf
import warnings
import numpy as np

from cleverhans.utils import _ArgsWrapper, create_logger

_logger = create_logger("cleverhans.utils.tf")

def model_eval_gan(
    sess,
    images,
    labels,
    predictions=None,
    predictions_rec=None,
    test_images=None,
    test_labels=None,
    feed=None,
    args=None,
    model=None,
    diff_op=None,
):
    """Computes the accuracy of a model on test data as well as the
    reconstruction errors for attack detection.
    
    Args:
        sess: TF session to use when training the graph.
        images: input placeholder.
        labels: output placeholder (for labels).
        predictions: model output predictions.
        predictions_rec: model output prediction for reconstructions.
        test_images: numpy array with training inputs
        test_labels: numpy array with training outputs
        feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
        args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
        model: (deprecated) if not None, holds model output predictions.
        diff_op: The operation that calculates the difference between input
            and attack.

    Returns:
        accuracy: The accuracy on the test data.
        accuracy_rec: The accuracy on the reconstructed test data (if
            predictions_rec is provided)
        roc_info: The differences between input and reconstruction for
            attack detection.
    """
    args = _ArgsWrapper(args or {})

    assert args.batch_size, "Batch size was not given in args dict"
    if test_images is None or test_labels is None:
        raise ValueError("X_test argument and Y_test argument "
                         "must be supplied.")
    if model is None and predictions is None:
        raise ValueError("One of model argument "
                         "or predictions argument must be supplied.")
    if model is not None:
        warnings.warn("model argument is deprecated. "
                      "Switch to predictions argument. "
                      "model argument will be removed after 2018-01-05.")
        if predictions is None:
            predictions = model
        else:
            raise ValueError("Exactly one of model argument"
                             " and predictions argument should be specified.")

    # Define accuracy symbolically.
    correct_preds = tf.equal(tf.argmax(labels, axis=-1),
                             tf.argmax(predictions, axis=-1))

    if predictions_rec is not None:
        correct_preds_rec = tf.equal(tf.argmax(labels, axis=-1),
                                     tf.argmax(predictions_rec, axis=-1))
        acc_value_rec = tf.reduce_sum(tf.to_float(correct_preds_rec))

    accuracy_rec = 0.0
    cur_labels = tf.argmax(labels, axis=-1),
    cur_preds = tf.argmax(predictions, axis=-1)

    acc_value = tf.reduce_sum(tf.to_float(correct_preds))


    diffs = []
    all_labels = []
    preds = []

    accuracy = 0.0

    # Compute number of batches.
    nb_batches = int(math.ceil(float(len(test_images)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(test_images)

    for batch in range(nb_batches):
        # To initialize the variables of Defense-GAN at test time.
        sess.run(tf.local_variables_initializer())
        print("[#] Eval batch {}/{}".format(batch, nb_batches))

        # Must not use the `batch_indices` function here, because it
        # repeats some examples.
        # It's acceptable to repeat during training, but not eval.
        start = batch * args.batch_size
        end = min(len(test_images), start + args.batch_size)
        cur_batch_size = end - start

        # The last batch may be smaller than all others, so we need to
        # account for variable batch size here.
        feed_dict = {images: test_images[start:end], labels: test_labels[start:end]}
        if feed is not None:
            feed_dict.update(feed)



        run_list = [acc_value,cur_labels,cur_preds]

        if diff_op is not None:
            run_list += [diff_op]

        if predictions_rec is not None:
            run_list += [acc_value_rec]
            acc_val_ind = len(run_list)-1;

        outs = sess.run(run_list,feed_dict=feed_dict)
        cur_acc = outs[0]

        if diff_op is not None:
            cur_diffs_val = outs[3]
            diffs.append(cur_diffs_val)

        if predictions_rec is not None:
            cur_acc_rec = outs[acc_val_ind]
            accuracy_rec += cur_acc_rec

        cur_labels_val = outs[1][0]
        cur_preds_val = outs[2]
        all_labels.append(cur_labels_val)
        preds.append(cur_preds_val)

        accuracy += cur_acc

    assert end >= len(test_images)

    # Divide by number of examples to get final value.
    accuracy /= len(test_images)
    accuracy_rec /= len(test_images)
    preds = np.concatenate(preds)
    all_labels = np.concatenate(all_labels)

    if diff_op is not None:
        diffs = np.concatenate(diffs)

    roc_info = [all_labels,preds,diffs]
    if predictions_rec is not None:
        return accuracy,accuracy_rec,roc_info
    else:
        return accuracy, roc_info
