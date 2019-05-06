import tensorflow as tf

from neural_toolbox import utils


def co_attention (question_states,description,history_states,image_feature,no_mlp_units,reuse=False):
    pass

def compute_all_attention(question_states,description,history_states,image_feature,no_mlp_units,reuse=False):
    print("image_feature = {}",image_feature)
    print("Question = {}",question_states)
    print("description = {}",description)
    print("history_sta = {}",history_states)

    ##### 1 ####
    # recupere les données

    #### 2 ####
    #step_dictionnaire

    ##### loop step_dictionnaire ########
    # feature_input = mlp(x,g1,g2) [1.. feature_shape]
    # soft_feature = softmax(feature_input) [1.. feature_input_shape]
    # x = soft_feature * x [1.. feature_input_shape]
    with tf.variable_scope("coattention"):

        if len(image_feature.get_shape()) == 3:
            h = tf.shape(image_feature)[1]  # when the shape is dynamic (attention over lstm)
            w = 1
            c = int(image_feature.get_shape()[2])
        else:
            h = int(image_feature.get_shape()[1])
            w = int(image_feature.get_shape()[2])
            c = int(image_feature.get_shape()[3])
     
        s = int(question_states.get_shape()[2])

        image_feature = tf.reshape(image_feature, shape=[-1, h * w, c]) # input image_feature ?,7,7,2048 => ?,49,2048
        question_states = tf.reshape(question_states,[-1,10,1024])

        ques = question_states
        img  = image_feature
        hist = tf.reshape(history_states,[-1,6,1024])

        description = tf.expand_dims(description,axis=1)
        hist = tf.concat([description,hist],axis=1)

        step_attention = {0:[img,ques,None],1:[hist,img,ques],2:[ques,img,hist],3:[img,hist,ques]}


        for key,value in step_attention.items():
            hidden_mlp,weight = utils.fully_connected(value[0], no_mlp_units, scope='hidden_layer', activation="tanh", reuse=reuse,
                                      co_attention=True,g1=value[1],g2=value[2])

            print("... Input_data = {} ".format(value[0]))
            print("... G1 = {} ".format(value[1]))
            print("... G2 = {} ".format(value[2]))
            exit()
            hidden_mlp = utils.fully_connected(hidden_mlp, 1, scope='out-onelayer', reuse=reuse,co_attention=False)
            
            print("img = {} ,hidden_mlp = {} ".format(value[0],hidden_mlp))
            # hidden_mlp = tf.reshape(hidden_mlp, shape=[-1, h * w, 2048])

            print("Input_softmax = ",hidden_mlp)
            
            # out = tf.matmul(hidden_mlp, weight)

            alpha = utils.fully_connected(hidden_mlp, no_mlp_units, scope='out-softmax', activation="softmax", reuse=reuse,co_attention=False)
            print(" alpha = ",alpha)
            
            value[0] = value[0] * alpha

            print("hidden = {} ,ALPHA= {} ,INPUT_DATA = {}".format(hidden_mlp,alpha,value[0]))

            exit()            


        


        print("Done ::")
        exit()
        
        
        # embedding = tf.concat([image_feature, question_states], axis=2) # shape=(?, 49, 3072)
        # embedding = tf.reshape(embedding, shape=[-1, s + c]) # shape=(?, 3072)




        # compute the evidence from the embedding
        with tf.variable_scope("mlp_1"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)
            print(" before ** e = {}".format(e))
            e = utils.fully_connected(e, 1, scope='out', reuse=reuse)

        print(" After ** e = {}".format(e))

        exit()


        e = tf.reshape(e, shape=[-1, h * w, 1])
        alpha = tf.nn.softmax(e,dim=1) # shape=(?, 49, 1)
        soft_attention = image_feature * alpha # shape=(?, 49, 2048)

        i_feature = tf.reduce_sum(soft_attention, axis=1) # shape=(?, 2048)

        #-------------------------------------- 2 -----------------------------------------
        q_feature = None
        h_feature = None



       
        

        all_history_res = tf.reshape(history_states,[6,1024])
        all_history = tf.concat([description,history_states],axis=1)




        with tf.variable_scope("mlp_1"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)
            e = utils.fully_connected(e, 1, scope='out', reuse=reuse,co_attention=True,g1=i_feature)
        
        e = tf.reshape(e, shape=[-1, h * w, 1])
        alpha = tf.nn.softmax(e,dim=1) # shape=(?, 49, 1)
        h_feature = all_history * alpha # shape=(?, 49, 2048)







        

    return soft_attention

    # def compute_all_attention(question_states,description,history_states,image_feature,no_mlp_units,reuse=False):
    # # print("image_feature = {}",image_feature)
    # # print("question = {}",question_states)
    # # print("description = {}",description)
    # # print("history_sta = {}",history_states)


    # with tf.variable_scope("coattention"):

    #     if len(image_feature.get_shape()) == 3:
    #         h = tf.shape(image_feature)[1]  # when the shape is dynamic (attention over lstm)
    #         w = 1
    #         c = int(image_feature.get_shape()[2])

    #     else:
    #         h = int(image_feature.get_shape()[1])
    #         w = int(image_feature.get_shape()[2])
    #         c = int(image_feature.get_shape()[3])
        
    #     # print("h={},w={},c={}".format(h,w,c))
    #     # exit()

    #     # print("shape = ",question_states.get_shape()[2])
    #     s = int(question_states.get_shape()[2])

    #     image_feature = tf.reshape(image_feature, shape=[-1, h * w, c]) # input image_feature ?,7,7,2048 => ?,49,2048

    #     # print("Reshape Feature_maps ={}".format(image_feature))
        

    #     # print("Reshape context ={}".format(contattention_mode2]) => [a,b,c,a,b,c]
    #     # print("Reshape context ={}".format(context))s=2)

    #     embedding = tf.concat([image_feature, question_states], axis=2)
    #     print(" +++ embedding = {} ".format(embedding))
        

    #     embedding = tf.reshape(embedding, shape=[-1, s + c])

    #     print("------- embedding = ",embedding)
    #     # exit()



    #     # compute the evidence from the embedding
    #     with tf.variable_scope("mlp"):
    #         e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)
    #         e = utils.fully_connected(e, 1, scope='out', reuse=reuse)

    #     print(" before ** e = {}".format(e))


    #     e = tf.reshape(e, shape=[-1, h * w, 1])
    #     print(" after ** e = {}".format(e))
        

    #     # compute the softmax over the evidenceprint(" image = {} ".format(self._crop))
    #             # print(" Crop = {} ".format(self.crop_out))
    #             # print(" co_attention = {} ".format(co_attention))
                
    #             # exit()
    #     # print(e)
    #     alpha = tf.nn.softmax(e,dim=1) # shape=(?, 49, 1)
        
    #     print(" Alpha ** = {} ".format(alpha))


    #     # first_attention = 



    #     # apply soft attention
    #     soft_attention = image_feature * alpha # shape=(?, 49, 2048)
    #     print(" soft_attention *B* = {} ".format(soft_attention))

    #     soft_attention = tf.reduce_sum(soft_attention, axis=1) # shape=(?, 2048)
    #     print(" soft_attention *AF* = {} ".format(soft_attention))

    #     # print(soft_attention)

    #     # print(soft_attention)
    #     exit()

    # return soft_attention








def compute_attention(feature_maps, context, no_mlp_units, reuse=False):

    print("****** Feature_maps ={},    context={}".format(feature_maps,context))
    # exit()

    # print("Feature_maps SHAPE={},context SHAPE={}".format(feature_maps.get_shape(),no_mlp_units.get_shape()))

    with tf.variable_scope("attention"):

        if len(feature_maps.get_shape()) == 3:
            h = tf.shape(feature_maps)[1]  # when the shape is dynamic (attention over lstm)
            w = 1
            c = int(feature_maps.get_shape()[2])
        else:
            h = int(feature_maps.get_shape()[1])
            w = int(feature_maps.get_shape()[2])
            c = int(feature_maps.get_shape()[3])

        s = int(context.get_shape()[1])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])#Tensor("oracle/attention/Reshape:0", shape=(?, 49, 2048), dtype=float32)

        print("Reshape Feature_maps ={}".format(feature_maps))

        context = tf.expand_dims(context, axis=1)#Tensor("oracle/attention/ExpandDims:0", shape=(?, 1, 6144), dtype=float32)
        print("Reshape context ={}".format(context))

        context = tf.tile(context, [1, h * w, 1]) # tf.tile([a,b,c],dimension=[2]) => [a,b,c,a,b,c]
                                                  # Tensor("oracle/attention/Tile:0", shape=(?, 49, 6144), dtype=float32)
        print("Tile context ={}".format(context))


        embedding = tf.concat([feature_maps, context], axis=2)
        embedding = tf.reshape(embedding, shape=[-1, s + c])#Tensor("oracle/attention/Reshape_1:0", shape=(?, 8192), dtype=float32)
        print("Embedding = ",embedding)


        # compute the evidence from the embedding
        with tf.variable_scope("mlp"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)#Tensor("oracle/attention/mlp/Relu:0", shape=(?, 256), dtype=float32)
            # print(" Before E = {}".format(e))
            e = utils.fully_connected(e, 1, scope='out', reuse=reuse) # Tensor("oracle/attention/mlp/out/add:0", shape=(?, 1), dtype=float32)
            # print(" After E = {}".format(e))

        e = tf.reshape(e, shape=[-1, h * w, 1])#Tensor("oracle/attention/Reshape_2:0", shape=(?, 49, 1), dtype=float32)

        # compute the softmax over the evidence
        print("Reshaphe e=",e)
        alpha = tf.nn.softmax(e,dim=1) # Tensor("oracle/attention/transpose_1:0", shape=(?, 49, 1), dtype=float32)
        print("alpha = {} , feature_map = {} ".format(alpha,feature_maps))



        # apply soft attention
        soft_attention = feature_maps * alpha # Tensor("oracle/attention/mul:0", shape=(?, 49, 2048), dtype=float32)
        print("soft_attention = ",soft_attention)

        soft_attention = tf.reduce_sum(soft_attention, axis=1) # Tensor("oracle/attention/Sum:0", shape=(?, 2048), dtype=float32)
        print("soft_attention reduce_sum = ",soft_attention)
        # exit()

    return soft_attention


# cf https://arxiv.org/abs/1610.04325

def compute_glimpse(feature_maps, context, no_glimpse, glimpse_embedding_size, keep_dropout, reuse=False):
    with tf.variable_scope("glimpse"):
        h = int(feature_maps.get_shape()[1])
        w = int(feature_maps.get_shape()[2])
        c = int(feature_maps.get_shape()[3])

        # reshape state to perform batch operation
        context = tf.nn.dropout(context, keep_dropout)
        projected_context = utils.fully_connected(context, glimpse_embedding_size,
                                                  scope='hidden_layer', activation="tanh",
                                                  use_bias=False, reuse=reuse)

        projected_context = tf.expand_dims(projected_context, axis=1)
        projected_context = tf.tile(projected_context, [1, h * w, 1])
        projected_context = tf.reshape(projected_context, [-1, glimpse_embedding_size])

        feature_maps = tf.reshape(feature_maps, shape=[-1, h * w, c])

        glimpses = []
        with tf.variable_scope("glimpse"):
            g_feature_maps = tf.reshape(feature_maps, shape=[-1, c])  # linearise the feature map as as single batch
            g_feature_maps = tf.nn.dropout(g_feature_maps, keep_dropout)
            g_feature_maps = utils.fully_connected(g_feature_maps, glimpse_embedding_size, scope='image_projection',
                                                   activation="tanh", use_bias=False, reuse=reuse)

            hadamard = g_feature_maps * projected_context
            hadamard = tf.nn.dropout(hadamard, keep_dropout)

            e = utils.fully_connected(hadamard, no_glimpse, scope='hadamard_projection', reuse=reuse)
            e = tf.reshape(e, shape=[-1, h * w, no_glimpse])

            for i in range(no_glimpse):
                ev = e[:, :, i]
                alpha = tf.nn.softmax(ev)
                # apply soft attention
                soft_glimpses = feature_maps * tf.expand_dims(alpha, -1)
                soft_glimpses = tf.reduce_sum(soft_glimpses, axis=1)

                glimpses.append(soft_glimpses)

        full_glimpse = tf.concat(glimpses, axis=1)

    return full_glimpse
