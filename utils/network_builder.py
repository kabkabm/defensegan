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

"""Modified for Defense-GAN:
- Added the ReconstructionLayer class for cleverhans.
- The different model architectures that are tested in the paper.

Modified version of cleverhans/model.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta

import keras.backend as K
import numpy as np
import tensorflow as tf


class Model(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """
        return self.get_probs(*args, **kwargs)

    def get_layer(self, x, layer):
        """
        Expose the hidden features of a model given a layer name.
        :param x: A symbolic representation of the network input
        :param layer: The name of the hidden layer to return features at.
        :return: A symbolic representation of the hidden features
        :raise: NoSuchLayerError if `layer` is not in the model.
        """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested

    def get_logits(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output logits (i.e., the
                 values fed as inputs to the softmax layer).
        """
        return self.get_layer(x, 'logits')

    def get_probs(self, x):
        """
        :param x: A symbolic representation of the network input
        :return: A symbolic representation of the output probabilities (i.e.,
                the output values produced by the softmax layer).
        """
        try:
            return self.get_layer(x, 'probs')
        except NoSuchLayerError:
            import tensorflow as tf
            return tf.nn.softmax(self.get_logits(x))

    def get_layer_names(self):
        """
        :return: a list of names for the layers that can be exposed by this
        model abstraction.
        """

        if hasattr(self, 'layer_names'):
            return self.layer_names

        raise NotImplementedError('`get_layer_names` not implemented.')

    def fprop(self, x):
        """
        Exposes all the layers of the model returned by get_layer_names.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError('`fprop` not implemented.')


class CallableModelWrapper(Model):

    def __init__(self, callable_fn, output_layer):
        """
        Wrap a callable function that takes a tensor as input and returns
        a tensor as output with the given layer name.
        :param callable_fn: The callable function taking a tensor and
                            returning a given layer as output.
        :param output_layer: A string of the output layer returned by the
                             function. (Usually either "probs" or "logits".)
        """

        self.output_layer = output_layer
        self.callable_fn = callable_fn

    def get_layer_names(self):
        return [self.output_layer]

    def fprop(self, x):
        return {self.output_layer: self.callable_fn(x)}


class NoSuchLayerError(ValueError):
    """Raised when a layer that does not exist is requested."""


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape, rec_model=None):
        super(MLP, self).__init__()
        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False, no_rec=False):
        states = []
        start = 0
        if no_rec:
            start = 1

        for layer in self.layers[start:]:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

    def add_rec_model(self, model, z_init, batch_size):
        rec_layer = ReconstructionLayer(model, z_init, self.input_shape, batch_size)
        rec_layer.set_input_shape(self.input_shape)
        self.layers = [rec_layer] + self.layers
        self.layer_names = ['reconstruction'] + self.layer_names


class Layer(object):
    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):
    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):
    def __init__(self, output_channels, kernel_shape, strides, padding):
        self.__dict__.update(locals())
        del self.self

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class ReconstructionLayer(Layer):
    """This layer is used as a wrapper for Defense-GAN's reconstruction
    part.
    """

    def __init__(self, model, z_init, input_shape, batch_size):
        """Constructor of the layer.

        Args:
            model: `Callable`. The generator model that gets an input and
                reconstructs it. `def gen(Tensor) -> Tensor.`
            z_init: `tf.Tensor'.
            input_shape: `List[int]`.
            batch_size: int.
        """
        self.z_init = z_init
        self.rec_model = model
        self.input_shape = input_shape
        self.batch_size = batch_size

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        x.set_shape(self.input_shape)
        self.rec = self.rec_model.reconstruct(
            x, batch_size=self.batch_size, back_prop=True, z_init_val=self.z_init,
            reconstructor_id=123)
        return self.rec


class ReLU(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)


class Dropout(Layer):
    def __init__(self, prob):
        self.prob = prob
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.cond(K.learning_phase(), lambda: tf.nn.dropout(x, self.prob), lambda: x)


class Softmax(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):
    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


def model_f(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_e(input_shape=(None, 28, 28, 1), nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """

    # Define a fully connected model (it's different than the black-box).
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def model_d(input_shape=(None, 28, 28, 1), nb_classes=10):
    """
    Defines the model architecture to be used by the substitute. Use
    the example model interface.
    :param img_rows: number of rows in input
    :param img_cols: number of columns in input
    :param nb_classes: number of classes in output
    :return: tensorflow model
    """

    # Define a fully connected model (it's different than the black-box)
    layers = [Flatten(),
              Linear(200),
              ReLU(),
              Dropout(0.5),
              Linear(200),
              ReLU(),
              Linear(nb_classes),
              Softmax()]

    return MLP(layers, input_shape)


def model_b(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Dropout(0.2),
              Conv2D(nb_filters, (8, 8), (2, 2), "SAME"),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID"),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID"),
              ReLU(),
              Dropout(0.5),
              Flatten(),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_a(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (5, 5), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Dropout(0.25),
              Linear(128),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_c(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (5, 5), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Dropout(0.25),
              Linear(128),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_y(nb_filters=64, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_q(nb_filters=32, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(256),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model


def model_z(nb_filters=32, nb_classes=10,
            input_shape=(None, 28, 28, 1), rec_model=None):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME"),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(2 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Conv2D(4 * nb_filters, (3, 3), (1, 1), "VALID"),
              ReLU(),
              Conv2D(4 * nb_filters, (3, 3), (2, 2), "VALID"),
              ReLU(),
              Flatten(),
              Linear(600),
              ReLU(),
              Dropout(0.5),
              Linear(600),
              ReLU(),
              Dropout(0.5),
              Linear(nb_classes),
              Softmax()]

    model = MLP(layers, input_shape, rec_model=rec_model)
    return model
