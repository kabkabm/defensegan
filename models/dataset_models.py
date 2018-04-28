import tensorflow as tf

import tflib as lib
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def mnist_generator(n_samples, noise=None, use_bn=False,
                    net_dim=64, output_dim=64, is_training=False,
                    latent_dim=128):
    if noise is None:
        noise = tf.random_normal([n_samples, latent_dim])

    output = lib.ops.linear.Linear('Generator.Input', latent_dim,
                                   4 * 4 * 4 * net_dim, noise)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output,
                                             is_training=is_training)

    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, 4, 4, 4 * net_dim, ])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * net_dim, 2 * net_dim,
                                       5, output)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 1, 2],
                                             output, is_training=is_training)
    output = tf.nn.relu(output)

    output = output[:, :7, :7, :]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * net_dim, net_dim, 5,
                                       output)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 1, 2],
                                             output, is_training=is_training)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', net_dim, 1, 5, output)
    output = tf.nn.sigmoid(output)

    return output


def mnist_discriminator(inputs, use_bn=False, net_dim=128, is_training=False):
    output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, net_dim, 5, inputs,
                                   stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', net_dim, 2 * net_dim, 5,
                                   output, stride=2)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 1, 2],
                                             output, is_training=is_training)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * net_dim, 4 * net_dim,
                                   5, output, stride=2)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 1, 2],
                                             output, is_training=is_training)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * net_dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * net_dim,
                                   1, output)

    return tf.reshape(output, [-1])


def MnistEncoder(inputs, use_bn=False, net_dim=128, is_training=False,
                 latent_dim=None):
    output = lib.ops.conv2d.Conv2D('Encoder.1', 1, net_dim, 5, inputs,
                                   stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Encoder.2', net_dim, 2 * net_dim, 5,
                                   output, stride=2)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Encoder.BN2', [0, 1, 2], output,
                                             is_training=is_training)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Encoder.3', 2 * net_dim, 4 * net_dim, 5,
                                   output, stride=2)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Encoder.BN3', [0, 1, 2], output,
                                             is_training=is_training)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * net_dim])
    output = lib.ops.linear.Linear('Encoder.Output', 4 * 4 * 4 * net_dim,
                                   latent_dim, output)

    return tf.tanh(output)


def celeba_generator(n_samples, noise=None, use_bn=False,
                     net_dim=64, output_dim=64, is_training=False,
                     latent_dim=128, stats_iter=None):
    if noise is None:
        noise = tf.random_normal([n_samples, latent_dim])

    output = lib.ops.linear.Linear('Generator.Input', latent_dim,
                                   4 * 4 * 4 * net_dim, noise)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output,
                                             is_training=is_training,
                                             stats_iter=stats_iter)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4, 4, 4 * net_dim])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4 * net_dim, 2 * net_dim,
                                       5, output)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0, 1, 2],
                                             output, is_training=is_training,
                                             stats_iter=stats_iter)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2 * net_dim, net_dim, 5,
                                       output)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0, 1, 2],
                                             output, is_training=is_training,
                                             stats_iter=stats_iter)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', net_dim, net_dim, 5,
                                       output)

    output = lib.ops.deconv2d.Deconv2D('Generator.6', net_dim, 3, 5, output)

    output = tf.tanh(output)

    return output


def celeba_discriminator(inputs, use_bn=False, net_dim=128, is_training=False,
                         stats_iter=None, data_format='NCHW'):
    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, net_dim, 5, inputs,
                                   stride=2, data_format=data_format)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', net_dim, 2 * net_dim, 5,
                                   output, stride=2,
                                   data_format=data_format)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0, 2, 3],
                                             output, is_training=is_training,
                                             stats_iter=stats_iter,
                                             data_format=data_format)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2 * net_dim, 4 * net_dim,
                                   5, output, stride=2,
                                   data_format=data_format)
    if use_bn:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0, 2, 3],
                                             output, is_training=is_training,
                                             stats_iter=stats_iter)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * net_dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * net_dim,
                                   1, output)

    return tf.reshape(output, [-1])
