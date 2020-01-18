# pylint: disable=wrong-import-position
import os
import sys

from tensorflow.python.client import device_lib  # pylint: disable=no-name-in-module
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def multiply_matrices():

    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        return sess.run(c)


def main():

    gpus = get_available_gpus()

    print()
    print('GPUs available')
    print('--------------')
    print(f'{gpus if gpus else "No GPUs available"}')
    if not gpus:
        sys.exit()

    print()
    print('Test matrix multiplication')
    print('--------------------------')
    print(multiply_matrices())


if __name__ == "__main__":
    main()
