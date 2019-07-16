import tensorflow as tf
from tensorflow.python.ops.init_ops import UniformUnitScaling, Constant
import logging

#TODO slowly delete those modules


def get_embedding(lookup_indices, n_words, n_dim,
                  scope="embedding", reuse=False,all_embedding = None,dict_all_embedding=[]):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.control_dependencies([tf.assert_non_negative(n_words - tf.reduce_max(lookup_indices))]):
            
            if len(dict_all_embedding) > 0:
                embedding_matrix = tf.constant(dict_all_embedding,shape=[len(dict_all_embedding),100],name='Ws',dtype=tf.float32)
                # embedding_matrix_2 = tf.get_variable('W', [n_words, n_dim],initializer=tf.constant_initializer(0.08))
                # print("embedding_matrix_1 = {} ".format(embedding_matrix_1))
                # print("embedding_matrix_2 = {} ".format(embedding_matrix_2))
                # exit()
            else:
                embedding_matrix = tf.get_variable('W', [n_words, n_dim],initializer=tf.constant_initializer(0.08))


            
            print("embedding_matrix = {}".format(embedding_matrix))
            embedded = tf.nn.embedding_lookup(embedding_matrix, lookup_indices)
            print("Utils | embedding={}".format(embedded))


            return embedded


def fully_connected(inp, n_out, activation=None, scope="fully_connected",
                    weight_initializer=UniformUnitScaling(),
                    init_bias=0.0, use_bias=True, reuse=False,co_attention=False,g1=None,g2=None,key_input=None):

    logger = logging.getLogger()
    with tf.variable_scope(scope, reuse=reuse):
        
        # print("--- inp_size = {} ".format(inp))

        inp_size = int(inp.get_shape()[1])

        shape = [inp_size, n_out]

        weight = tf.get_variable(
        "W", shape,
        initializer=weight_initializer)


        print("-- INPUTS_SHAPE = {} , WEIGTH = {} ".format(inp,weight))
        out = tf.matmul(inp, weight)

        
       
        if co_attention:
            # inp_size = inp.get_shape()[1:3]
            # shape = [int(inp_size[0]),int(inp_size[1]), n_out]
            # weight = tf.get_variable(
            # "W", shape,
            # initializer=weight_initializer)

            # print(" Input = {}".format(inp))
            # print("input_size = {},weigth = {} ".format(inp_size,weight))
            
            out = tf.matmul(inp, weight)
        
            if g1 != None and g2 != None:
                inp_size = int(g1.get_shape()[1])
                shape = [inp_size, n_out]

                print("shape = {}".format(shape))
                weight_g1 = tf.get_variable(
                "W2", shape,
                initializer=weight_initializer)

                inp_size = int(g2.get_shape()[1])
                shape = [inp_size, n_out]
                weight_g2 = tf.get_variable(
                "W3", shape,
                initializer=weight_initializer)

                out_1 = tf.matmul(g1, weight_g1)
                out_2 = tf.matmul(g2, weight_g2)
                # print(" out_put_1 = {} ,out_put_2 = {} ".format(out_1,out_2))

            if g1!= None and g2==None:
                # inp_size = g1.get_shape()[1:3]
                # shape = [int(inp_size[0]),int(inp_size[1]), n_out]
                # # print(" g1 = {}".format(g1))
                # # print("input_size = {} ".format(inp_size))
                # weight_g1 = tf.get_variable(
                # "W2", shape,
                # initializer=weight_initializer)

                inp_size = int(g1.get_shape()[1])
                shape = [inp_size, n_out]
                weight_g1 = tf.get_variable(
                "W1", shape,
                initializer=weight_initializer)
                # out = tf.matmul(inp, weight)
                out_1 = tf.matmul(g1, weight_g1)

           

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
          

                # out = tf.reshape(out,[out_1Dim*out_2Dim,out_last_dim])
                # out_1 = tf.reshape(out_1,[out1_1Dim*out1_2Dim,out1_last_dim])
                # out_2 = tf.reshape(out_2,[out1_2Dim*out2_2Dim,out2_last_dim])



                # print("-----3out- out = {} , out_1 = {},out_2 = {}".format(out,out_1,out_2))

                all_ouput = [out,out_1,out_2] # Tensor("oracle/coattention/concat_1:0", shape=(2501, 256), dtype=float32)
                # print("------ all_ouput = {} ".format(all_ouput))

                # print("********* out = {} ".format(out))
                # print("********* out_1 = {} ".format(out_1))
                # print("********* out_2 = {} ".format(out_2))

                sum_ouput = tf.add_n(all_ouput)
                # print("------ sum_ouput = {} ".format(sum_ouput))
            
                out_tanh = tf.tanh(sum_ouput)
                # print("------ out_tanh = {} ".format(out_tanh))

            if g1!=None and g2==None:
            

                # out = tf.reshape(out,[out_1Dim*out_2Dim,out_last_dim])
                # out_1 = tf.reshape(out_1,[out1_1Dim*out1_2Dim,out1_last_dim])
                # print("------ out = {} , out_1 = {}".format(out,out_1))

                all_ouput = [out,out_1] # Tensor("oracle/coattention/concat_1:0", shape=(2501, 256), dtype=float32)
                # print("------ all_ouput = {} ".format(all_ouput))
                # print("********* out = {} ".format(out))
                # print("********* out_1 = {} ".format(out_1))
                

                sum_ouput = tf.add_n(all_ouput)
                # print("------ sum_ouput_None = {} ".format(sum_ouput))
            
                out_tanh = tf.tanh(sum_ouput)
                # print("------ out_tanh = {} ".format(out_tanh))
                




        return out_tanh,weight
    return out

def rank(inp):
    return len(inp.get_shape())


def cross_entropy(y_hat, y):
    if rank(y) == 2:
        # print("cross_entropy -------- y_glod = {} y_predict = {} ".format(y_hat,y))
        # -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
        # print("cross_entropy = {}",-tf.reduce_sum(y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0))))
        # exit()
        # y_pred = tf.constant([0.3,0.7,0.0],dtype=tf.float32)
        # tf.reduce_mean(y*tf.log(tf.clip_by_value(y_hat,1e-10,1.0)))

        return -tf.reduce_mean(y*tf.log(y_hat))

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
