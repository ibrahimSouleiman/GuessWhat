import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.resnet_v1 as resnet_v1
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


def create_resnet(image_input, is_training, scope="",scope_feature="", resnet_out="logits", resnet_version=50, cbn=None):
    """
    Create a resnet by overidding the classic batchnorm with conditional batchnorm
    :param image_input: placeholder with image
    :param is_training: are you using the resnet at training_time or test_time
    :param scope: tensorflow scope
    :param resnet_version: 50/101/152
    :param cbn: the cbn factory
    :return: the resnet output
    """

    # print("resnet_out = {}".format(resnet_out))

    if cbn is None:
        # assert False, "\n" \
        #               "There is a bug with classic batchnorm with slim networks (https://github.com/tensorflow/tensorflow/issues/4887). \n" \
        #               "Please use the following config -> 'cbn': {'use_cbn':true, 'excluded_scope_names': ['*']}"
        # arg_sc = slim_utils.resnet_arg_scope(is_training=is_training)
        
        arg_sc = slim_utils.resnet_arg_scope()
        # print("arg_sc = {}".format(arg_sc))
    else:
        arg_sc = get_resnet_arg_scope(cbn.apply)
     

    # Pick the correct version of the resnet
    if resnet_version == 50:
        print("------ 50")
        current_resnet = resnet_v1.resnet_v1_50
    elif resnet_version == 101:
        print("------ 101")
        current_resnet = resnet_v1.resnet_v1_101
    elif resnet_version == 152:
        print("------ 152")
        current_resnet = resnet_v1.resnet_v1_152
    else:
        raise ValueError("Unsupported resnet version")

    resnet_scope = os.path.join('resnet_v1_{}/'.format(resnet_version), resnet_out)
    # print(" resnet_out = {} , resnet_scope = {}".format(resnet_out,resnet_scope))

    # exit()
    # print("current_resnet = {}".format(current_resnet))
    # exit()

    with slim.arg_scope(arg_sc):
        net, end_points = current_resnet(image_input, 1000,scope="resnet_v1_50")  # 1000 is the number of softmax class
    
    print("net = {}, end_points = {}".format(net,end_points))



    # print(" resnet | endpoint=",end_points)
    if len(scope) > 0 and not scope.endswith("/"):
        scope += "/"

 

    out = end_points[scope +  resnet_scope] # Tensor("oracle/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0", shape=(32, 7, 7, 2048), dtype=float32) 

  
    # print("------------------------- out Use: {},output = {}".format(resnet_scope,out))
    # out = tf.reshape(
    # out,
    # [-1,out.shape[3]],
    # )

    # print("-- net = {}, end_points={},out={} ".format(net,end_points,out))
    # exit()

    return out,end_points
