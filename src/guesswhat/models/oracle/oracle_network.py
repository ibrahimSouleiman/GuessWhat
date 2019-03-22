import tensorflow as tf

from neural_toolbox import rnn, utils

from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features

class OracleNetwork(ResnetModel):

    def __init__(self, config, num_words_question ,num_words_description,  device='', reuse=False):
        ResnetModel.__init__(self, "oracle", device=device)

        with tf.variable_scope(self.scope_name, reuse=reuse) as scope:
            embeddings = []
            self.batch_size = None

            # QUESTION
            if config['inputs']['question']:
                print("----------------------------------------")
                print("**** Oracle_network |  inpurt = question ")

                self._is_training = tf.placeholder(tf.bool, name="is_training")
                self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
                self.seq_length_question = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_question')

                word_emb = utils.get_embedding(self._question,
                                            n_words=num_words_question,
                                            n_dim=int(config['model']['question']["embedding_dim"]),
                                            scope="word_embedding")



                if config['embedding'] != "None":
                
                    self._glove = tf.placeholder(tf.float32, [None, None, 100], name="embedding_vector")
                    word_emb = tf.concat([word_emb, self._glove], axis=2)
                else:
                    print("None -------------------------- None")
		  
                
                lstm_states, _ = rnn.variable_length_LSTM(word_emb,
                                                    num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                    seq_length=self.seq_length_question)
                

                embeddings.append(lstm_states)

            # DESCRIPTION
            if config['inputs']['description']:
                print("****  Oracle_network |  inpurt = Description ")

                self._is_training = tf.placeholder(tf.bool, name="is_training")
                self._description = tf.placeholder(tf.int32, [self.batch_size, None], name='description')
                self.seq_length_description = tf.placeholder(tf.int32, [self.batch_size], name='seq_length_description')

                word_emb = utils.get_embedding(self._description,
                                            n_words=num_words_description,
                                            n_dim=int(config['model']['description']["embedding_dim"]),
                                            scope="word_embedding_description")

                #print(" SeqDescription = ",self.seq_length_description)
                lstm_states_description, _ = rnn.variable_length_LSTM(word_emb,
                                                    num_hidden=int(config['model']['question']["no_LSTM_hiddens"]),
                                                    seq_length=self.seq_length_description,scope="lstm2")
                embeddings.append(lstm_states_description)


                

            # CATEGORY
            if config['inputs']['category']:
                print("****  Oracle_network |  input = category ")

                self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')

                cat_emb = utils.get_embedding(self._category,
                                              int(config['model']['category']["n_categories"]) + 1,  # we add the unkwon category
                                              int(config['model']['category']["embedding_dim"]),
                                              scope="cat_embedding")

                


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

                self._image = tf.placeholder(tf.float32, [self.batch_size] + config['model']['image']["dim"], name='image')
                self.image_out = get_image_features(
                    image=self._image, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config['model']['image']
                )
                embeddings.append(self.image_out)
                print("Input: Image")

            # CROP
            if config['inputs']['crop']:
                print("****  Oracle_network |  input = crop ")

                self._crop = tf.placeholder(tf.float32, [self.batch_size] + config['model']['crop']["dim"], name='crop')
                self.crop_out = get_image_features(
                    image=self._crop, question=lstm_states,
                    is_training=self._is_training,
                    scope_name=scope.name,
                    config=config["model"]['crop'])

                embeddings.append(self.crop_out)
                print("Input: Crop")


            # Compute the final embedding
            emb = tf.concat(embeddings, axis=1)

            # OUTPUT
            num_classes = 3
            self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')

            with tf.variable_scope('mlp'):
                num_hiddens = config['model']['MLP']['num_hiddens']
                l1 = utils.fully_connected(emb, num_hiddens, activation='relu', scope='l1')

                self.pred = utils.fully_connected(l1, num_classes, activation='softmax', scope='softmax')
                self.best_pred = tf.argmax(self.pred, axis=1)

            self.loss = tf.reduce_mean(utils.cross_entropy(self.pred, self._answer))
            self.error = tf.reduce_mean(utils.error(self.pred, self._answer))

            print('Model... Oracle build!')
            # print(" Summary = ",tf.summary())

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return 1. - self.error
