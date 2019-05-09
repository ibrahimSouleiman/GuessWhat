import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant
import logging

#TODO slowly delete those modules


def get_embedding(lookup_indices, n_words, n_dim,
                  scope="embedding", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.control_dependencies([tf.assert_non_negative(n_words - tf.reduce_max(lookup_indices))]):
            embedding_matrix = tf.get_variable(
                'W', [n_words, n_dim],
                initializer=tf.random_uniform_initializer(-0.08, 0.08))
            embedded = tf.nn.embedding_lookup(embedding_matrix, lookup_indices)
            print("Utils | embedding={}".format(embedded))
            return embedded


def fully_connected(inp, n_out, activation=None, scope="fully_connected",
                    weight_initializer=UniformUnitScaling(),
                    init_bias=0.0, use_bias=True, reuse=False,co_attention=False,g1=None,g2=None,key_input=None):
    logger = logging.getLogger()
    with tf.variable_scope(scope, reuse=reuse):
       
        if co_attention:
            inp_size = inp.get_shape()[1:3]
            shape = [inp_size[0],inp_size[1], n_out]
            weight = tf.get_variable(
            "W", shape,
            initializer=weight_initializer)

            print(" Input = {}".format(inp))
            print("input_size = {},weigth = {} ".format(inp_size,weight))
            


            if key_input == 0:
                logger.info('')
                out = tf.matmul(inp, weight)
            elif key_input == 1:
                out = tf.matmul(inp, weight)
            elif key_input == 2:
                out = tf.matmul(inp, weight)
            elif key_input == 3:
                out = tf.matmul(inp, weight)

            


            if g1 != None and g2 != None:
                inp_size = g1.get_shape()[1:3]
                shape = [inp_size[0],inp_size[1], n_out]

                print("shape = {}".format(shape))
                weight_g1 = tf.get_variable(
                "W2", shape,
                initializer=weight_initializer)

                inp_size = g2.get_shape()[1:3]
                shape = [inp_size[0],inp_size[1], n_out]
                weight_g2 = tf.get_variable(
                "W3", shape,
                initializer=weight_initializer)

                out_1 = tf.matmul(g1, weight_g1)
                out_2 = tf.matmul(g2, weight_g2)
                # print(" out_put_1 = {} ,out_put_2 = {} ".format(out_1,out_2))

            if g1!= None and g2==None:
                inp_size = g1.get_shape()[1:3]
                # print(" g1 = {}".format(g1))
                # print("input_size = {} ".format(inp_size))
                shape = [inp_size[0],inp_size[1], n_out]
                weight_g1 = tf.get_variable(
                "W2", shape,
                initializer=weight_initializer)

                out_1 = tf.matmul(g1, weight_g1)

        else:
            inp_size = int(inp.get_shape()[1])
            shape = [inp_size, n_out]
            weight = tf.get_variable(
            "W-1", shape,
            initializer=weight_initializer)
            out = tf.matmul(inp, weight)

            # print("**** input = {} , inp_size = {} ,shape = {}, weigth = {} ,out = {} ".format(inp,inp_size,shape,weight,out))





        if use_bias:
             bias = tf.get_variable(
                 "b", [n_out],
                 initializer=Constant(init_bias))
             out += bias

    if activation == 'relu':
        return tf.nn.relu(out)
    if activation == 'softmax':
        # shape = [inp,n_out]
        # weight = tf.get_variable(
        #     "W_softmax", shape,
        #     initializer=weight_initializer)

        # print("**** weigth = {} ,weigth_transpose = {} ".format(weight,tf.transpose(weight)))
        # out = tf.matmul(tf.transpose(weight),inp)

        return tf.nn.softmax(out)

        # print("out fully_connected= {}".format(out))
        # print(tf.nn.softmax(out))
        # pass
       

       
    if activation == 'tanh':
        out_tanh = tf.tanh(out)

        if co_attention:
            if g1!=None and g2!=None:
                out_1Dim = out.get_shape()[0]
                out_2Dim = out.get_shape()[1]
                out_last_dim = out.get_shape()[2]

                out1_1Dim = out_1.get_shape()[0]
                out1_2Dim = out_1.get_shape()[1]
                out1_last_dim = out.get_shape()[2]


                out2_1Dim = out_2.get_shape()[0]
                out2_2Dim = out_2.get_shape()[1]
                out2_last_dim = out.get_shape()[2]

                # out = tf.reshape(out,[out_1Dim*out_2Dim,out_last_dim])
                # out_1 = tf.reshape(out_1,[out1_1Dim*out1_2Dim,out1_last_dim])
                # out_2 = tf.reshape(out_2,[out1_2Dim*out2_2Dim,out2_last_dim])



                # print("-----3out- out = {} , out_1 = {},out_2 = {}".format(out,out_1,out_2))

                all_ouput = tf.concat([out,out_1,out_2],axis=1) # Tensor("oracle/coattention/concat_1:0", shape=(2501, 256), dtype=float32)
                # print("------ all_ouput = {} ".format(all_ouput))

                sum_ouput = tf.reduce_sum(all_ouput,1)
                # print("------ sum_ouput = {} ".format(sum_ouput))
            
                out_tanh = tf.tanh(sum_ouput)
                # print("------ out_tanh = {} ".format(out_tanh))

            if g1!=None and g2==None:
                out_1Dim = out.get_shape()[0]
                out_2Dim = out.get_shape()[1]
                out_last_dim = out.get_shape()[2]

                out1_1Dim = out_1.get_shape()[0]
                out1_2Dim = out_1.get_shape()[1]

                out1_last_dim = out.get_shape()[2]

                # out = tf.reshape(out,[out_1Dim*out_2Dim,out_last_dim])
                # out_1 = tf.reshape(out_1,[out1_1Dim*out1_2Dim,out1_last_dim])

                # print("------ out = {} , out_1 = {}".format(out,out_1))

                all_ouput = tf.concat([out,out_1],axis=1) # Tensor("oracle/coattention/concat_1:0", shape=(2501, 256), dtype=float32)
                # print("------ all_ouput = {} ".format(all_ouput))

                sum_ouput = tf.reduce_sum(all_ouput,1)
                # print("------ sum_ouput = {} ".format(sum_ouput))
            
                out_tanh = tf.tanh(sum_ouput)
                # print("------ out_tanh = {} ".format(out_tanh))
                




        return out_tanh,weight
    return out

def rank(inp):
    return len(inp.get_shape())


def cross_entropy(y_hat, y):
    if rank(y) == 2:
        
        return -tf.reduce_mean(y * tf.log(y_hat))
    if rank(y) == 1:
        ind = tf.range(tf.shape(y_hat)[0]) * tf.shape(y_hat)[1] + y
        flat_prob = tf.reshape(y_hat, [-1])

        

        return -tf.log(tf.gather(flat_prob, ind))
    raise ValueError('Rank of target vector must be 1 or 2')


def error(y_hat, y):

    if rank(y) == 1:
        mistakes = tf.not_equal(
            tf.argmax(y_hat, 1), tf.cast(y, tf.int64))
    elif rank(y) == 2:
        mistakes = tf.not_equal(
            tf.argmax(y_hat, 1), tf.argmax(y, 1))
    else:
        assert False
    return tf.cast(mistakes, tf.float32)
