Distributed Trace Analysis LSTM algorithm
=========================================


Run
---
.. code-block:: console

    cd dta_lstm
    python -m tests.functional.run_dta_lstm


Configuration
-------------

The file `config.py` is used to configure STrace LSTM

* `$ vi iforesight2/algorithms/strace_lstm/config.py`

.. code-block:: python

    DEFAULTS = {
        'threshold': .7,
        'top_k': 4,
        'epochs': 100,
        'batch_size': 50,
        'validation_split': 0.15,
        'explain': True,
        'gpus': 1,
        'span_list_col_name': 'span_list',
    }


Tensorflow
----------

Install tensorflow.

+ CentOS 7
+ python 3.6
+ Uninstall tensorflow and install tensorflow-gpu (Keras selects the cpu version by default)
+ nvidia card (check card: $ nvidia-smi -l 1)
+ read https://www.tensorflow.org/install/pip

.. code-block:: console

    pip uninstall tensorflow tensorflow-gpu
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.12.0-cp36-cp36m-linux_x86_64.whl
    nvidia-smi -l 1


Test your GPU:

.. code-block:: python

    import tensorflow as tf
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    with tf.Session() as sess:
        print (sess.run(c))

The output is:

.. code-block:: console

    2019-02-06 02:29:25.182359: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
    2019-02-06 02:29:26.903779: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1432] Found device 0 with properties:
    name: Tesla M60 major: 5 minor: 2 memoryClockRate(GHz): 1.1775
    pciBusID: 0000:85:00.0
    totalMemory: 7.93GiB freeMemory: 7.85GiB
    2019-02-06 02:29:26.903914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
    2019-02-06 02:29:27.258389: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
    2019-02-06 02:29:27.258462: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0
    2019-02-06 02:29:27.258476: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N
    2019-02-06 02:29:27.258784: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7576 MB memory) -> physical GPU (device: 0, name: Tesla M60, pci bus id: 0000:85:00.0, compute capability: 5.2)

    [[22. 28.]
     [49. 64.]]


To enable more than one GPU to be used, the environment variable CUDA_VISIBLE_DEVICES needs to specify which
GPUs ids are visible. In your python code, ad the following lines:

.. code-block:: python

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

