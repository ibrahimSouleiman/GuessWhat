import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
import tensorflow.contrib.slim.python.slim.nets.inception_v1 as inception_v1

import tensorflow.contrib.slim.python.slim.nets.resnet_utils as slim_utils

from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
import os

def get_resnet_arg_scope(bn_fn):
    """
    Trick to apply CBN from a pretrained tf network. It overides the batchnorm constructor with cbn
    :param bn_fn: cbn factory
    :return: tensorflow scope
    """

    with arg_scope(
            [layers_lib.conv2d],
            activation_fn=tf.nn.relu,
            normalizer_fn=bn_fn,
            normalizer_params=None) as arg_sc:
        return arg_sc


def create_inception(image_input, is_training, scope="", inception_out="Mixed_5c", resnet_version=50, cbn=None):
    """
    Create a resnet by overidding the classic batchnorm with conditional batchnorm
    :param image_input: placeholder with image
    :param is_training: are you using the resnet at training_time or test_time
    :param scope: tensorflow scope
    :param resnet_version: 50/101/152
    :param cbn: the cbn factory
    :return: the resnet output
    """

        # assert False, "\n" \
        #               "There is a bug with classic batchnorm with slim networks (https://github.com/tensorflow/tensorflow/issues/4887). \n" \
        #               "Please use the following config -> 'cbn': {'use_cbn':true, 'excluded_scope_names': ['*']}"
        # arg_sc = slim_utils.resnet_arg_scope(is_training=is_training)
    
    # print("--- 1") 
    arg_sc = inception_v1.inception_v1_arg_scope()

    # Pick the correct version of the resnet
    # if resnet_version == 50:
    #     current_resnet = resnet_v1.resnet_v1_50
    # elif resnet_version == 101:
    #     current_resnet = resnet_v1.resnet_v1_101
    # elif resnet_version == 152:
    #     current_resnet = resnet_v1.resnet_v1_152
    # else:
    #     raise ValueError("Unsupported resnet version")

    # inception_scope = os.path.join('InceptionV1/InceptionV1', inception_out)
    # print("--- 2")
    inception_scope = inception_out
    # print(" resnet_out = {} , resnet_scope = {}".format(resnet_out,resnet_scope))
    # print("--- 3")
    with slim.arg_scope(arg_sc):
        net, end_points = inception_v1.inception_v1(image_input, 1001)  # 1000 is the number of softmax class


    print("Net = ",net)
    # print("--- 4")
    
    if len(scope) > 0 and not scope.endswith("/"):
        scope += "/"
    # print("--- 5")
    # print(end_points)
    print(" Batch ",inception_scope)

    out = end_points[scope + inception_scope]
    print("-- out Use: {},output = {}".format(inception_scope,out))

    return out,end_points
