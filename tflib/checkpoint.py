import os
import tensorflow as tf

def save_model(saver,sess,checkpoint_dir, step,model_name="GAN.model"):
    '''
    Saves to output
    :param checkpoint_dir:
    :param step:
    :return:
    '''
    saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

import re

def load_model(saver,sess,checkpoint_dir=None):
        '''
        Loads the saved model
        :param checkpoint_dir: root of all the checkpoints
        :return:
        '''
        FLAGS = tf.app.flags.FLAGS

        def load_from_path(ckpt_path):
            ckpt_name = os.path.basename(ckpt_path)
            saver.restore(sess, ckpt_path)
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        try:
            if ckpt and ckpt.model_checkpoint_path:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
                return load_from_path(os.path.join(checkpoint_dir, ckpt_name))
            else:
                print(" [*] Failed to find a checkpoint within directory {}".format(FLAGS.ckpt_path))
                return False, 0
        except Exception as e:
            print(e)
            print(" [*] Failed to find a checkpoint, Exception!")
            return False, 0
