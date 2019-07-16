import tensorflow as tf


# For some reason, it is faster than MultiCell on tf



def lstm(cell,rnn_states,seq_length):
    rnn_states, rnn_last_states = tf.nn.dynamic_rnn(
                        cell,
                        rnn_states,
                        dtype=tf.float32,
                        sequence_length=seq_length,
                        )
    return rnn_states

def variable_length_LSTM(inp, num_hidden, seq_length,
                         dropout_keep_prob=1.0, scope="lstm", depth=1,
                         layer_norm=False, reuse=False,dim_4=False):

    with tf.variable_scope(scope, reuse=reuse):
        states = []
        last_states = []
        # rnn_states = inp
        for d in range(depth):
            with tf.variable_scope('lstmcell'+str(d)):

                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                    num_hidden,
                    layer_norm=layer_norm,
                    dropout_keep_prob=dropout_keep_prob,
                    reuse=reuse)

                if dim_4: 

                    for i in range(len(inp)):
                        # print("rnn ques = {} sequence_length = {}".format(inp[i],seq_length[i]))
                        rnn_states, rnn_last_states = tf.nn.dynamic_rnn(
                            cell,
                            inp[i],
                            dtype=tf.float32,
                            sequence_length=seq_length[i],
                            )

                        # print("rnn h_last= {} , h_all = {} ".format(rnn_states,rnn_last_states)) # shape=(?, ?, 1024),
                        # print("type = {} ".format(type(rnn_states)))
                        states.append(rnn_states)
                        last_states.append(rnn_last_states.h)         
                                                           
                else:

                    # print("****** inp =  {} , seq_length = {} ".format(inp,seq_length))
                        
                    rnn_states, rnn_last_states = tf.nn.dynamic_rnn(
                        cell,
                        inp,
                        dtype=tf.float32,
                        sequence_length=seq_length,
                    )

                    # print("All param inp = {} , num_hidden = {} , seq_length = {} ".format(inp,num_hidden,seq_length))

                    # print("** rnn_states = {} ,last_states = {} , .h= {} ".format(rnn_states,
                    #                                                     rnn_last_states,rnn_last_states.h))
                    states.append(rnn_states)
                    last_states.append(rnn_last_states.h)                    
                    states = tf.concat(states, axis=0)
                    

                    # print("states = {} ".format(states))
                    # print("last_states = {} ".format(last_states))
                    # exit()


                 


                    
                    # last_states = tf.concat(last_states, axis=1)
                    # print("AFTER Last_states = {} ".format(last_states))
                    # exit()

        if dim_4:
            print("BEFORE states  ,last_states = {} ".format(last_states))

        last_states = tf.concat(last_states, axis=1)

        if dim_4:
            print("** LSTM OUTPUT =",last_states)
            print("AFTER states  ,last_states = {} ".format(last_states))
            

        return last_states, states



