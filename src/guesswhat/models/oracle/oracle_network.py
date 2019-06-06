import tensorflow as tf

from neural_toolbox import rnn, utils

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features

from neural_toolbox.attention import compute_all_attention

class OracleNetwork(ResnetModel):

    def __init__(self, config, num_words_question ,num_words_description=None,  device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            embeddings = []
            co_attention = [None,None,None,None]
            self.batch_size = None

            # QUESTION
            if config['inputs']['question']:
     
                self._is_training = tf.placeholder(tf.bool, name="is_training")
                self._question = tf.placeholder(tf.int32, [self.batch_size, 6], name='question')
                self.seq_length_question = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_question')

                word_emb = utils.get_embedding(self._question,
                                            n_words=num_words_question,
                                            n_dim=int(config["model"]["word_embedding_dim"]),
                                            scope="word_embedding")

                if config["model"]["glove"] == True or config["model"]["fasttext"] == True:
                    self._embWord = tf.placeholder(tf.float32, [self.batch_size, 10, int(config["model"]["word_embedding_dim"])], name="embedding_vector_ques")
                    word_emb = self._embWord

                    # word_emb = tf.concat([ word_emb,self._embWord], axis=2)
                else:
                    print("None -------------------------- None")
		  
                #     if config['glove']:
                # self._glove = tf.placeholder(tf.float32, [None, None, int(config['model']['question']["embedding_dim"])], name="glove")
                # word_emb = tf.concat([word_emb, self._glove], axis=2)
                    

                print("word_emb = {} ".format(word_emb))


                self.out_question = None

                if config['model']['question']['lstm']:
                    self.lstm_states, self.lstm_all_state_ques = rnn.variable_length_LSTM(word_emb,
                                                        num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                        seq_length=self.seq_length_question)
                    self.out_question =  self.lstm_all_state_ques
                else:
                    self.out_question = word_emb


                

                if config["model"]["attention"]["co-attention"]:
                    co_attention[0] = self.out_question     # embeddings.append(self.lstm_all_state_ques)
                else:
                    embeddings.append(self.out_question)


                # print("*+*+*+*+* **+*+*+**Length=",self.seq_length_question)
                # print("word_emb = {}".format(word_emb))
                # print("question = {}".format(self._question))
                # print("lstm_all_state_ques = {} ".format(self.lstm_all_state_ques))
                # exit()



                # QUESTION-Pos

                if config['model']['question'] ['pos']:
                    print("----------------------------------------")
                    print("**** Oracle_network |  input = question-pos ")

                    self._question_pos = tf.placeholder(tf.int32, [self.batch_size, None], name='question_pos')
                    self.seq_length_pos = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_ques_pos')

                    word_emb = utils.get_embedding(self._question_pos,
                                                n_words=num_words_question,
                                                n_dim=300,
                                                scope="word_embedding_pos")

                    if config["model"]["glove"] == True or config["model"]["fasttext"] == True:
                        self._glove = tf.placeholder(tf.float32, [None, None,int(config["model"]["word_embedding_dim"])], name="embedding_vector_ques_pos")
                        word_emb = tf.concat([word_emb, self._glove], axis=2)

                    else:
                        print("None ****************")
                    
                    lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                                                        num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                        seq_length=self.seq_length_pos,scope="lstm2")
                    


                    # embeddings.append(lstm_states)

            # DESCRIPTION

        

            if config['inputs']['description']:
                print("****  Oracle_network |  inpurt = Description ")

                self._description = tf.placeholder(tf.int32, [self.batch_size, None], name='description')
                self.seq_length_description = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_description')

                # word_emb = utils.get_embedding(self._description,
                #                             n_words=num_words_description,
                #                             n_dim=300,
                #                             scope="word_embedding_description")
                
                if config["model"]["glove"] == True or config["model"]["fasttext"] == True:
                    self._glove = tf.placeholder(tf.float32, [None, None,int(config["model"]["word_embedding_dim"])], name="embedding_vector_des")
                    word_emb =  self._glove

                    # print("Word_emb",word_emb)
                    # exit()
                else:
                    print("None ****************")

                

                print(" SeqDescription = ",self.seq_length_description)


                # if config['model']['question']['lstm']:
                lstm_states_description, _ = rnn.variable_length_LSTM(word_emb,
                                        num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                        seq_length=self.seq_length_description,scope="lstm3")
                self.out_description = lstm_states_description
                # else:
                #     self.out_description = word_emb
                
                if config["model"]["attention"]["co-attention"]:
                    history = []
                    history.append(self.out_description)
                    # self._question_history = tf.placeholder(tf.int32, [self.batch_size, None , None,None], name='question_history')

                    if config["model"]["glove"] == True or config["model"]["fasttext"] == True:
                        placeholders_lstmQuestion = []
                        placeholders_lstmLength = []

                        self._embWord = tf.placeholder(tf.float32, [None,None,None, int(config["model"]["word_embedding_dim"])], name="embedding_vector_ques_hist")
                        # self.seq_length_question_history = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_question_history')

                        for i in range(6):
                            self._embWord = tf.placeholder(tf.float32, [None,None, int(config["model"]["word_embedding_dim"])], name="embedding_vector_ques_hist_H{}".format(i))
                            self.seq_length_question_history = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_question_history_H{}'.format(i))
                            

                            placeholders_lstmQuestion.append(self._embWord)
                            placeholders_lstmLength.append(self.seq_length_question_history)
                       
                        # shape = [batch_size,history_length,max_length_sentences,dim_embedding]
                        word_emb = self._embWord                    
                    else:
                        print("None -------------------------- None")

              


                    print("*+*+*+*+*+*+*+**Length=",self.seq_length_question_history,word_emb)
                    
                    print("Batch_size = ",self.batch_size)



                    self.lstm_states, self.lstm_all_state_ques_hist = rnn.variable_length_LSTM(placeholders_lstmQuestion,
                                                    num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                    seq_length=placeholders_lstmLength,scope="lstm4",dim_4=True)





                if config["model"]["attention"]["co-attention"]:
                    history.append(self.lstm_all_state_ques_hist)
                    # print("History = {}".format(history))
                    print("-**** lstm_states_description = ",self.out_description)
                    # lstm_states_description = tf.expand_dims(lstm_states_description,1)
                    # print(lstm_states_description)
                    # exit()

                    co_attention[1] = self.out_description
                    co_attention[2] = self.lstm_states


                    # embeddings.append(lstm_states_description)
                    # embeddings.append(self.lstm_states)

                else:
                    embeddings.append(self.out_description)

                
                # embeddings.append(lstm_states_description )


             # Description-Pos

                if config['model']['description'] ['pos']:
                    print("----------------------------------------")
                    print("**** Oracle_network |  inpurt = question-pos ")

                    self._question_pos = tf.placeholder(tf.int32, [self.batch_size, None], name='des_pos')
                    self.seq_length_pos = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_des_pos')

                    word_emb = utils.get_embedding(self._question_pos,
                                                n_words=num_words_question,
                                                n_dim=300,
                                                scope="word_embedding_pos")

                    if config["model"]["glove"] == True or config["model"]["fasttext"] == True:
                        self._glove = tf.placeholder(tf.float32, [None, None, int(config["model"]["word_embedding_dim"])], name="embedding_vector_des_pos")
                        word_emb = tf.concat([word_emb, self._glove], axis=2)
                    else:
                        print("None ****************")
                    
                    lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                                                        num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                        seq_length=self.seq_length_pos,scope="lstm5")
                    

                    # embeddings.append(lstm_states)

            # CATEGORY
            if config['inputs']['category']:
                print("****  Oracle_network |  input = category ")

                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')


                print(".... ORACLE =",self._category)

                cat_emb = utils.get_embedding(self._category,
                                              int(config['model']['category']["n_categories"]) + 1,  # we add the unkwon category
                                              int(config["model"]["word_embedding_dim"]),
                                              scope="cat_embedding")

                

                # cat_emb = tf.expand_dims(cat_emb,1)
                embeddings.append(cat_emb)
                print("Input: Category")


            # ALLCATEGORY
            if config['inputs']['allcategory']:
                print("**** Oracle_network |  input = allcategory ")

        
                
                self._allcategory = tf.placeholder(tf.float32, [self.batch_size,90], name='allcategory')
                # self.seq_length_allcategory = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_allcategory')

                # word_emb = utils.get_embedding(self._allcategory,
                #                             n_words=int(config['model']['category']["n_categories"]) + 1,
                #                             n_dim=int(config['model']['description']["embedding_dim"]),
                #                             scope="word_embedding_allcategory")

                
                #print(" SeqDescription = ",self.seq_length_description)
                # lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                #                                     num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                #                                     seq_length=self.seq_length_allcategory,scope="lstm3")
                
                print(" Oracle_network | embdedding all_cate=",word_emb)
                embeddings.append(self._allcategory)
                print("Input: allcategory")

                
            # SPATIAL
            if config['inputs']['spatial']:
                print("****  Oracle_network |  input = spatial ")
                self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
                embeddings.append(self._spatial)
                print("Input: Spatial")


            # IMAGE
            if config['inputs']['image']:
                print("****  Oracle_network |  input = image ")

                # self._image_id = tf.placeholder(tf.float32, [self.batch_size], name='image_id')

                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['model']['image']["dim"], name='image')
                self.image_out = get_image_features(
                    image=self._image, question=self.lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['model']['image']
                )

                # embeddings.append(self.image_out)
                print("Input: Image")
                co_attention[3]  = self.image_out

                print("... Image Features = {}".format(self.image_out))

            # CROP
            if config['inputs']['crop']:
                print("****  Oracle_network |  input = crop ")
                # self._image_id = tf.placeholder(tf.float32, [self.batch_size], name='image_id')
                # self._crop_id = tf.placeholder(tf.float32, [self.batch_size], name='crop_id')

                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['model']['crop']["dim"], name='crop')

                self.crop_out = get_image_features(
                    image=self._crop, question=self.lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config["model"]['crop'])

                co_attention[3] = self.crop_out

      
            question_feature,history_feature,image_feature = compute_all_attention(question_states=co_attention[0],
                                                                                caption=co_attention[1],
                                                                                history_states=co_attention[2],
                                                                                image_feature=co_attention[3],
                                                                                no_mlp_units=config['model']['attention']['no_attention_mlp'],
                                                                                config = config
                                                                                )


            print("image_feature = ",image_feature)
            print("question_feature = ",question_feature)
            print("history_feature = ",history_feature)

          

           


            embeddings.append(image_feature)
            embeddings.append(history_feature)
            embeddings.append(question_feature)


            # embeddings.append(question_feature)
            # print("*** All Embedding = ",embeddings)
            # exit()

            self.emb = tf.concat(embeddings, axis=1)
            
            print("*** self.emb = ",self.emb)
            


            # Compute the final embedding
            # print("---------- Embeddings=",embeddings)
            # self.emb = tf.concat(embeddings, axis=1)
            
        

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')



            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                # emb = tf.print(emb, [emb], "input: ")
                l1 = utils.fully_connected(self.emb, num_hiddens, activation='relu', scope='l1')
                self.pred = utils.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                self.best_pred = tf.argmax(self.pred, axis=1)
            # self.best_pred = tf.reduce_mean(self.best_pred)

            print("--- predict = {} ,answer = {} ".format(self.pred,self._answer))
            
            # self.loss = None
            self.loss = tf.reduce_mean(utils.cross_entropy(self.pred, self._answer))
            self.error = tf.reduce_mean(utils.error(self.pred, self._answer))

            print("loss = {} ,error = {} ".format(self.loss,self.error))

            

            print('Model... Oracle build!')
            # print(" Summary = ",tf.summary())

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error

    def get_predict(self):
        return self.pred
