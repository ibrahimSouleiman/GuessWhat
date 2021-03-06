import tensorflow as tf

from neural_toolbox import utils


img = None
question = None
history = None


def get_img():
    global img
    return img

def get_question():
    global question
    return question

def get_history():
    global history
    return history


def set_img(new_img):
    global img
    img = new_img
    return img

def set_question(new_question):
    global question 
    question = new_question
    return question

def set_history(new_history):
    global history
    history = new_history
    return history




def co_attention (question_states,caption,history_states,image_feature,no_mlp_units,reuse=False):
    pass

def get_input_g1_g2(num_data,num_g1,num_g2):

    input_data = g1 = g2 = None

    print("*** get_input_g1_g2 = {},{},{} ".format(num_data,num_g1,num_g2))
    if num_data == 0:
        input_data = get_img()
    elif num_data == 1:
        input_data = get_question()
    elif num_data == 3:
        input_data = get_history()

    if num_g1 == 0:
        g1 = get_img()
    elif num_g1 == 1:
        g1 = get_question()
    elif num_g1 == 3:
        g1 = get_history()

    if num_g2 == 0:
        g2 = get_img()
    elif num_g2 == 1:
        g2 = get_question()
    elif num_g2 == 3:
        g2 = get_history()



    return input_data,g1,g2


def compute_all_attention(question_states,caption,history_states,image_feature,no_mlp_units,reuse=False,config=None):
    
    
    print("image_feature = {}",image_feature)
    print("Question = {}",question_states)
    print("caption = {}",caption)
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
        if image_feature != None:
            if len(image_feature.get_shape()) == 3:
                h = tf.shape(image_feature)[1]  # when the shape is dynamic (attention over lstm)
                w = 1
                c = int(image_feature.get_shape()[2])
            else:
                h = int(image_feature.get_shape()[1])
                w = int(image_feature.get_shape()[2])
                c = int(image_feature.get_shape()[3])
        
            s = int(question_states.get_shape()[2])
            image_feature = tf.reshape(image_feature, shape=[-1, (h * w) , c]) # input image_feature ?,7,7,2048 => ?,49,2048
            print("******************** B Image_feature = {} ".format(image_feature))
            image_feature = tf.reduce_sum(image_feature, axis=1)
            print("******************** A Image_feature = {} ".format(image_feature))


            set_img(image_feature)
        
        question_shape = question_states.get_shape()
        
        question_states = tf.reshape(question_states,shape=[-1,int(question_shape[1])*int(question_shape[2])])

   
        set_question(question_states)
       
        if history_states!= None:
            if caption != None:
                caption = tf.expand_dims(caption,axis=1)
                
            
            if caption != None:
                print("caption = {}  ,history_states = {} ".format(caption,history_states))
                history_states = tf.reshape(history_states,[-1,6,1024])
                hist = tf.concat([caption,history_states],axis=1)
                print("hist B= {} ".format(hist))
                hist = tf.reshape(history_states,[-1,7*1024])
               
            else:
                # hist = tf.reshape(history_states,[-1,6*1024])
                hist = history_states
                # print("hist = {} ".format(hist))
                # exit()
  
           
            set_history(hist)
       
        
        
        # return question_states,hist,image_feature

        dict_step = {0:"img",1:"question",3:"hist"}


        step_attention = {0:[0,1,None],1:[3,0,1],2:[1,0,3],3:[0,3,1]}

        # step_attention = {0:[0,1,None]}


        for key,value in step_attention.items():
            
            input_data , g1 , g2 = get_input_g1_g2(value[0],value[1],value[2])
            
            dimension_two = int(input_data.get_shape()[1])

            print("---- input_shape = {} ".format(input_data.get_shape()))
            print("---- g1_shape = {} ".format(g1.get_shape()))
            # print("g2_shape = {} ".format(g2.get_shape()))
  
            hidden_mlp,weight = utils.fully_connected(input_data, no_mlp_units, scope='hidden_layer_256_{}'.format(key), activation="tanh", reuse=reuse,
                                      co_attention=True,g1=g1,g2=g2,key_input=key)

            hidden_mlp = utils.fully_connected(hidden_mlp, 1, scope='hidden_layer_1_{}'.format(key), reuse=reuse,co_attention=False)
            

            alpha = tf.nn.softmax(hidden_mlp,axis=1)
            input_data = input_data * alpha
            if value[0] == 0:set_img(input_data)
            elif value[0] == 1:set_question(input_data)
            elif value[0] == 2:set_history(input_data)



            # print("-- {} -- hidden = {} ,ALPHA= {} ,INPUT_DATA = {}".format(key,hidden_mlp,alpha,input_data))
            # print("Data_ouput = ",get_img(),get_question(),get_history())
            # if key == 2:
            #     exit()
        
        # img_shape = get_img().get_shape()
        # question_shape = get_question().get_shape()
        # history_shape = get_history().get_shape()

        # img = tf.reshape(get_img(), shape=[-1,int(img_shape[1]) * int(img_shape[2]) ]) 
        # question = tf.reshape(get_question(), shape=[-1, int(question_shape[1]) * int(question_shape[2]) ]) 
        # history = tf.reshape(get_history(), shape=[-1,int(history_shape[1]) * int(history_shape[2]) ]) 
        question_states = get_question()
        history = get_history()
        image_feature = get_img() 

        # print(" history = {} , image_feature = {} , question = {}".format(history,image_feature,question_states))
        # exit()
               
        


    return question_states,history,image_feature

                     










def compute_attention(feature_maps, context, no_mlp_units, reuse=False):


  
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


        context = tf.expand_dims(context, axis=1) # Tensor("oracle/attention/ExpandDims:0", shape=(?, 1, 6144), dtype=float32)

        context = tf.tile(context, [1, h * w, 1]) # tf.tile([a,b,c],dimension=[2]) => [a,b,c,a,b,c]
                                                  # Tensor("oracle/attention/Tile:0", shape=(?, 49, 6144), dtype=float32)


        embedding = tf.concat([feature_maps, context], axis=2)
        embedding = tf.reshape(embedding, shape=[-1, s + c])#Tensor("oracle/attention/Reshape_1:0", shape=(?, 8192), dtype=float32)


        # compute the evidence from the embedding
        with tf.variable_scope("mlp"):
            e = utils.fully_connected(embedding, no_mlp_units, scope='hidden_layer', activation="relu", reuse=reuse)#Tensor("oracle/attention/mlp/Relu:0", shape=(?, 256), dtype=float32)
            # print(" Before E = {}".format(e))
            e = utils.fully_connected(e, 1, scope='out', reuse=reuse) # Tensor("oracle/attention/mlp/out/add:0", shape=(?, 1), dtype=float32)
            # print(" After E = {}".format(e))

        e = tf.reshape(e, shape=[-1, h * w, 1])#Tensor("oracle/attention/Reshape_2:0", shape=(?, 49, 1), dtype=float32)

        # compute the softmax over the evidence
        alpha = tf.nn.softmax(e,dim=1) # Tensor("oracle/attention/transpose_1:0", shape=(?, 49, 1), dtype=float32)



        # apply soft attention
        soft_attention = feature_maps * alpha # Tensor("oracle/attention/mul:0", shape=(?, 49, 2048), dtype=float32)

        soft_attention = tf.reduce_sum(soft_attention, axis=1) # Tensor("oracle/attention/Sum:0", shape=(?, 2048), dtype=float32)


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
