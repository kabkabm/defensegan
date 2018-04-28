
# coding: utf-8

# In[1]:


import sys
sys.path.append('../')
import _init_paths


# In[2]:


import time
from pip.utils import ensure_dir

import _init_paths

import cPickle
import numpy as np
from six.moves import xrange

import logging
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils import to_categorical
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_train
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks_tf import jacobian_graph, jacobian_augmentation

from models.gan import MnistDefenseGAN, CifarGANBase, FmnistDefenseDefenseGAN, CelebaGAN
from utils.config import load_config
from utils.gan_defense import model_eval_gan
from utils.network_builder import model_a, model_b, model_c, model_d, model_e, model_f, model_z
from utils.visualize import save_images_files
import keras.backend as K
from sklearn.neighbors import KNeighborsClassifier

import math
from cleverhans.utils import batch_indices, _ArgsWrapper, create_logger, set_log_level

import os
from datasets.celeba import CelebA
from datasets.dataset import PickleLazyDataset
from blackbox import get_cached_gan_data

FLAGS = flags.FLAGS
ds_gan = {'mnist': MnistDefenseGAN, 'cifar': CifarGANBase, 'f-mnist': FmnistDefenseDefenseGAN, 'celeba' : CelebaGAN, 'celeba_wider' : CelebaGAN}
orig_data_paths = {k: 'data/cache/{}_pkl'.format(k) for k in ds_gan.keys()}


import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds', required=True, help='Config file')
    parser.add_argument('--num_train', required=True, help='Config file',type=int)
    parser.add_argument('--sub', required=True, help='Config file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args, _ = parser.parse_known_args()
    return args






vectorize = lambda x : x.reshape([x.shape[0],-1])
def inv_onehot(ys):
    assert len(np.shape(ys)) > 1
    return np.argmax(ys,axis=1)

def convert_to_onehot(ys):
    max_y = int(np.max(ys))
    y_one_hat = np.zeros([len(ys), max_y + 1], np.float32)
    for (i, y) in enumerate(ys):
        y_one_hat[i, int(y)] = 1.0
    return y_one_hat


# In[10]:

args = parse_args()
if args.ds == 'mnist':
    from datasets.mnist import Mnist
    ds_train = Mnist()
    ds_val = Mnist()
elif args.ds == 'fmnist':
    from datasets.fmnist import FMnist
    ds_train = FMnist()
    ds_val = FMnist()

ds_train.load()

ds_val.load()

# In[9]:


print(ds_train.X.shape)
ds_train.X = ds_train.X / 255.0
ds_val.X = ds_val.X / 255.0
print('[#] Max: {}, Min: {}'.format(ds_train.X.max(),ds_train.X.min()))


# In[ ]:


num_train = args.num_train
num_tests = 2000
vectorize = lambda x : x.reshape([x.shape[0],-1])

X_train, Y_train, X_test, Y_test = [vectorize(ds_train.X[:num_train]), ds_train.y[:num_train], vectorize(ds_val.X[:num_tests]), ds_val.y[:num_tests]]


# In[ ]:


test_id = 0
Y_plt = Y_test[test_id]
X_plt = X_test[test_id]
#plt.imshow(X_plt.reshape((28,28))*255.0,cmap='gray')
print('[*] Label : {}'.format(Y_plt))


# In[6]:


model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,Y_train)
y_t = model.predict(vectorize(X_test))
accuracy = np.mean(y_t == Y_test)
print(Y_test)
print(y_t)
print(accuracy)
#knn_model = prep_bbox(X_train,Y_train,X_test,Y_test)


# In[7]:


sub_model_letter = args.sub
holdout = 150

x_shape, classes = list(X_train.shape[1:]), 10
models = {'A': model_a, 'B': model_b, 'C': model_c, 'D': model_d, 'E': model_e, 'F': model_f,
              'Z': model_z}

x_shape = [28,28,1]
x = tf.placeholder(tf.float32, shape=[None] + x_shape)
y = tf.placeholder(tf.float32, shape=(None, classes))

def to_mnist_shape(XX):
    return XX.reshape([XX.shape[0],28,28,1])

Y_test_ = convert_to_onehot(Y_test)
Y_train_ = convert_to_onehot(Y_train)

X_sub = X_test[:holdout]
Y_sub = Y_test[:holdout]
Y_sub_ = Y_test_[:holdout]
X_test = X_test[holdout:]
Y_test = Y_test[holdout:]

sub_model = models[sub_model_letter](input_shape = [None]+ x_shape,nb_classes = classes)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# In[ ]:


vectorize = lambda x : x.reshape([x.shape[0],-1])
def inv_onehot(ys):
    assert len(np.shape(ys)) > 1
    return np.argmax(ys,axis=1)

def convert_to_onehot(ys):
    max_y = int(np.max(ys))
    y_one_hat = np.zeros([len(ys), max_y + 1], np.float32)
    for (i, y) in enumerate(ys):
        y_one_hat[i, int(y)] = 1.0
    return y_one_hat


# In[ ]:


print session


# In[ ]:



def model_eval(sess, x, y,x_adv, model, X_test=None, Y_test=None,
               feed=None, args=None,debug=False):
    """
    Compute the accuracy of a TF model on some data
    :param sess: TF session to use when training the graph
    :param x: input placeholder
    :param y: output placeholder (for labels)
    :param predictions: model output predictions
    :param X_test: numpy array with training inputs
    :param Y_test: numpy array with training outputs
    :param feed: An optional dictionary that is appended to the feeding
             dictionary before the session runs. Can be used to feed
             the learning phase of a Keras model for instance.
    :param args: dict or argparse `Namespace` object.
                 Should contain `batch_size`
    :param model: (deprecated) if not None, holds model output predictions
    :return: a float with the accuracy value
    """
    args = _ArgsWrapper(args or {})
    
    accuracy = 0.0
    
    nb_batches = int(math.ceil(float(len(X_test)) / args.batch_size))
    assert nb_batches * args.batch_size >= len(X_test)
    to_digit = lambda x : np.argmax(x,axis=1)
    for batch in range(nb_batches):
        start = batch * args.batch_size
        end = min(len(X_test), start + args.batch_size)
        cur_batch_size = end - start
        feed_dict = {x: X_test[start:end], y: Y_test[start:end]}
        if feed is not None:
            feed_dict.update(feed)
        cur_X_test = sess.run(x_adv,feed_dict=feed_dict)
        if debug:
            debug_dir = 'debug/knn'
            ensure_dir(debug_dir)
            save_images_files(cur_X_test, output_dir=debug_dir, postfix='adv')
            save_images_files(X_test[start:end], output_dir=debug_dir, postfix='orig')
            raise ValueException("DEBUG")

        cur_preds = model.predict(cur_X_test.reshape([cur_X_test.shape[0],-1]))
        cur_acc = np.mean(cur_preds == inv_onehot(Y_test[start:end]))
        accuracy += (cur_batch_size * cur_acc)

        sys.stdout.write('\r [-] Eval batch {}/{}: acc: {}'.format(batch,nb_batches,accuracy*1.0/((batch+1)*args.batch_size+1)))
        sys.stdout.flush()


    assert end >= len(X_test)

    # Divide by number of examples to get final value
    accuracy /= len(X_test)
    sys.stdout.write('\r [*] Done with testing \n')

    return accuracy


# In[ ]:


sess = session
knn_model = model
nb_classes = classes
data_aug = 6
rng = np.random.RandomState([2017, 8, 30])
model_sub = sub_model
preds_sub = model_sub(x)
lmbda = 0.1

X_sub = X_test[:holdout]
Y_sub = Y_test[:holdout]

# Define the Jacobian symbolically using TensorFlow
grads = jacobian_graph(preds_sub, x, nb_classes)

# Train the substitute and augment dataset alternatively
for rho in xrange(data_aug):
    train_params = {
        'nb_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001
    }
    model_train(sess, x, y, preds_sub, to_mnist_shape(X_sub), convert_to_onehot(Y_sub),
                init_all=False, verbose=False, args=train_params,
                rng=rng, feed={K.learning_phase(): 1})

    # If we are not at last substitute training iteration, augment dataset
    if rho < data_aug - 1:
        print("Augmenting substitute training data.")
        # Perform the Jacobian augmentation
        X_sub = jacobian_augmentation(sess, x, to_mnist_shape(X_sub), Y_sub.astype(np.int), grads, lmbda, feed={K.learning_phase(): 0})

        print("Labeling substitute training data.")
        # Label the newly generated synthetic points using the black-box
        Y_sub = np.hstack([Y_sub, Y_sub])
        X_sub_prev = X_sub[int(len(X_sub) / 2):]
        yy = knn_model.predict(vectorize(X_sub_prev))
        Y_sub[int(len(X_sub) / 2):] = yy


# In[ ]:


eps = 0.3

fgsm_par = {'eps': eps, 'ord': np.inf, 'clip_min': 0., 'clip_max': 1.}
fgsm = FastGradientMethod(model_sub, sess=sess)

# Craft adversarial examples using the substitute
eval_params = {'batch_size': 32}
x_adv_sub = fgsm.generate(x, **fgsm_par)

diff = None

x_test = x_adv_sub
accuracy = model_eval(sess, x, y,x_adv_sub, model, to_mnist_shape(X_test), convert_to_onehot(Y_test),
        args=eval_params, feed={K.learning_phase(): 0},debug=False)
print('Test accuracy of oracle on adversarial examples generated '
          'using the substitute: ' + str(accuracy))

