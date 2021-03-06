import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier
from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image
from generic.data_provider.nlp_utils import padder,padder_3d,padder_4d
from generic.data_provider.nlp_utils import Embeddings,get_embeddings
from gensim.models import word2vec,FastText,KeyedVectors
from src.guesswhat.data_provider.generate_categoryQuestion import Generate_Category
from matplotlib import pyplot as plt
from PIL import Image

import numpy
import time
import os

# TODO corriger erreur lié a embedding aleatoire des mot sans fasttext ou glove
answer_dict = \
    {  'Yes': np.array([1, 0, 0], dtype=np.int32),
       'No': np.array([0, 1, 0], dtype=np.int32),
       'N/A': np.array([0, 0, 1], dtype=np.int32)
    }

class OracleBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer_question, sources,glove=None,tokenizer_description = None ,embedding=None, status=list(),args=None,config=None,trainset=None):
        self.tokenizer_question = tokenizer_question
        self.tokenizer_description = tokenizer_description
        self.sources = sources
        self.status = status
        self.config = config
        embedding_name = ""

        if config["model"]["fasttext"]:
            embedding_name = "fasttext"

        elif config["model"]["glove"]:
            embedding_name = "glove"

        #   ted_en-20160408.zip
        #   all_question_game.txt

        ##### Embedding_type #####################
        # 0 : trained only question
        # 1 : pretained with wikipedia data_set
        # 2 : pretained + trained question 
        #
        ##########################################
        
        if config["model"]["question"]["embedding_type"] == 0:
            self.embedding = Embeddings(file_name=["all_question_game.txt" ], embedding_type = config["model"]["question"]["embedding_type"] ,emb_dim=100)
        
        elif config["model"]["question"]["embedding_type"] == 1:
            self.embedding = Embeddings(file_name=["ted_en-20160408.zip" ], embedding_type = config["model"]["question"]["embedding_type"] ,emb_dim=100)
        
        elif config["model"]["question"]["embedding_type"] == 2:
            self.embedding = Embeddings(file_name=["ted_en-20160408.zip","all_question_game.txt" ], embedding_type = config["model"]["question"]["embedding_type"], emb_dim=100)


        self.model_embedding = self.embedding.model


        # if self.config["model"]["fasttext"] : 
        #     pass
        #     # self.generate_category = Generate_Category(self.model_word,"fasttext",self.tokenizer_question,config["model"]["category"]["all_word"])

        # elif self.config["model"]["glove"]:
        #     self.glove = glove
        #     self.generate_category = Generate_Category(self.glove,"glove",self.tokenizer_question,config["model"]["category"]["all_word"])

        
    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games


    def apply(self, games):
        sources = self.sources
        
        batch = collections.defaultdict(list)
        batch_size = len(games)
        assert batch_size > 0


        for i, game in enumerate(games):
            batch['raw'].append(game)
            image = game.image
            
            if 'question' in sources :
                question = self.tokenizer_question.apply(game.questions[0],use_dict_ques=False) 
                # print("+++++ words_question = {} ".format(question))
   
                sp_zeros = np.zeros((14))
                sp_zeros [0:len(question)] = question

                batch["question"].append(sp_zeros)
                batch["seq_length_question"].append(len(question))





            if 'embedding_vector_ques' in sources:
                assert  len(game.questions) == 1
                # Add glove vectors (NB even <unk> may have a specific glove)
                # print("oracle_batchifier | question = {}".format(game.questions[0]))

                words = self.tokenizer_question.apply(game.questions[0],tokent_int=False)
                
                if "question_pos" in sources:
                    # print("/////////// question_pos")
                    embedding_vectors,embedding_pos = get_embeddings(words,pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos) # slow (copy gloves in process)
                    # print("..... question_pos............. embedding_vectors",len(embedding_vectors[0]))
                    batch['embedding_vector_ques'].append(embedding_vectors)
                    batch['embedding_vector_ques_pos'].append(embedding_pos)
                    batch['question_pos'].append(question)

                else:
                    embedding_vectors = self.embedding.get_embedding(words)

                    # print("embedding = {}".format(np.asarray(embedding_vectors).shape  ))
                    # exit()

                    # if "embedding_vector_ques" not in batch:
                    #     batch['embedding_vector_ques'] = np.zeros((batch_size,7,100))
                    batch['embedding_vector_ques'].append(embedding_vectors)
                   

            if 'description' in sources:

                description = self.tokenizer_question.apply(game.image.description,use_dict_ques=False) 
                # print("+++++ words_question = {} ".format(question))

                batch["description"].append(description)


            if 'ques_hist_H0' in sources:
                assert  len(game.questions) == 1
                

                # description = self.tokenizer_description.apply(game.image.description)

                # batch['description'].append(description)


                for j in range(6):
                    question_answer = game.all_last_question[0]
                    words = []

                    if len(question_answer) > 1: 
                        word = self.tokenizer_question.apply(game.all_last_question[0][1][0])
                        words = word
                    else:
                        word = self.tokenizer_question.apply(game.all_last_question[0][0])
                        words = word
                    
                    

                    sp_zeros = np.zeros((14))
                    # print("words = {} ".format(words))
                    sp_zeros [0:len(words)] = words
                    # print("sp_zeros = {} ".format(sp_zeros))    

                    batch['ques_hist_H{}'.format(j)].append(sp_zeros)
                    batch['seq_length_question_history_H{}'.format(j)].append(len(words))


            # print('embedding_vector_des'in sources)
            if 'embedding_vector_des'in sources:
                description = self.tokenizer_description.apply(game.image.description,tokent_int=False)

                #print("*************** Description =",description)
                # batch['description'].append(description)
                if "des_pos" in sources:
                    embedding_vectors,embedding_pos = get_embeddings(description,pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos) # slow (copy gloves in process)
                    batch['embedding_vector_des'].append(embedding_vectors)
                    batch['embedding_vector_des_pos'].append(embedding_pos)
                    # batch['des_pos'].append(question)

                else:
                    if self.config["model"]["fasttext"] : 
                        #print("++++++----- ++++++++ Dans fasttext ")
                        embedding_vectors,_ = get_embeddings(description,pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos) # slow (copy gloves in process)
                    elif self.config["model"]["glove"] : 
                        #print("++++++----- ++++++++ Dans glove ")
                        embedding_vectors = self.glove.get_embeddings(description)

                    # print("------ ELSE".format(embedding_vectors))
                    # exit()
                    batch['embedding_vector_des'].append(embedding_vectors)


            if 'answer' in sources:

                if "answer" not in batch:
                    batch["answer"] = np.zeros((batch_size,3))   


                # print("game.amswer = {}".format(game.answers))

                # exit()
                assert len(game.answers) == 1
                batch['answer'][i] = answer_dict[game.answers[0]]
                #print(" Correct Answer = ",game.answers[0])

            if 'category' in sources:
                use_embedding_cat = self.config["model"]["category"]["use_embedding"]

                if "category" not in batch:
                    if use_embedding_cat:
                        batch['category'] = np.zeros((batch_size,100))
                    else:
                        batch['category'] = np.zeros((batch_size))
                
                if use_embedding_cat:
                    embc = np.asarray(self.embedding.get_embedding([game.object.category]))
                    # embc = self.tokenizer_question.apply(game.object.category,use_dict_ques=False)
                    category_input = embc.reshape((100))
                else:
                    category_input = game.object.category_id
                  
                # print("category = {} ".format(category_input))
                batch['category'][i] = category_input

    

            if 'allcategory' in sources:
                allcategory = []
                allcategory_hot = np.zeros(shape=(90),dtype=int)
                # print("Oracle_batchifier |  Allcategory -------------------------------")

                for obj in game.objects:
                    allcategory.append(obj.category_id - 1)

                allcategory_hot[allcategory] = 1
                batch['allcategory'].append(allcategory_hot)

            if 'spatial' in sources:
                if 'spatial' not in batch:
                    batch['spatial'] = np.zeros((batch_size,8),dtype=float)
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'][i]  = spat_feat

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())
                batch['image_id'].append(image.get_idimage())
                # batch['crop_id'].append(game.object_id)
                # print("crop_id=",game.object.get_crop().shape)
                # exit()
                
            if 'image' in sources:
                features_image = image.get_image()
                batch['image'].append(features_image)
                batch['image_id'].append(image.get_idimage())
          

            if 'mask' in sources:
                assert "image" in batch['image'], "mask input require the image source"
                mask = game.object.get_mask()
                ft_width, ft_height = batch['image'][-1].shape[1],\
                                     batch['image'][-1].shape[2] # Use the image feature size (not the original img size)
                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['mask'].append(mask)

        # padding = self.embedding.get_embeddings(["<padding>"])[0]
        # print("padding | = {}".format(padding))

        # pad the questions
        
        
        
        # if "question" in sources:
        #     batch['question'] , batch['seq_length_question'] = padder(batch['question'],max_seq_length=14)


        if "question_pos" in sources:
            batch['question_pos'], batch['seq_length_ques_pos'] = padder(batch['question_pos'],
                                                            padding_symbol=self.tokenizer_question.padding_token)



        if "description" in sources:
            batch['description'], batch['seq_length_description'] = padder(batch['description'] )
            

        # batch['embedding_vector_pos'], _ = padder_3d(batch['embedding_vector_pos'])

        if 'embedding_vector_ques' in sources:
                        batch['embedding_vector_ques'],s = padder_3d(batch['embedding_vector_ques'],max_seq_length=12)
               

        if 'embedding_vector_ques_hist' in sources:
                        # print("Shape=",np.asarray(batch['embedding_vector_ques_hist'] ).shape)
                        batch_hist, size_sentences,max_seq = padder_4d(batch['embedding_vector_ques_hist'],max_seq_length=14)
                        batch_hist = np.asarray(batch_hist)
                        size_sentences = np.asarray(size_sentences)

                        batch['embedding_vector_ques_hist'] = batch_hist

                        for i in range(6):
                            batch['ques_hist_H{}'.format(i)] = batch_hist[:,i,:]
                            batch['seq_length_question_history_H{}'.format(i)] = size_sentences[:,i]

                        

                        #print("Len=",len(batch['seq_length_question']))


        if 'embedding_vector_ques_pos' in sources:
                    batch['embedding_vector_ques_pos'], _ = padder_3d(batch['embedding_vector_ques_pos'])

        
        if 'embedding_vector_des' in sources:
                    batch['embedding_vector_des'], batch['seq_length_description'] = padder_3d(batch['embedding_vector_des'])


        if 'embedding_vector_des_pos' in sources:
                    batch['embedding_vector_des_pos'], _ = padder_3d(batch['embedding_vector_des_pos'])




        # if 'description' in sources:
        #     # complete par padding en prenons la taille maximal
        # batch['description'], batch['seq_length_description'] = padder_3d(batch['description'])
        
        # print(" Bath = {} ".format(batch.keys()))
        # exit()
        # print("finish oracle_bachifier .... time=",total)
        # print("TotalBatch=",total)


        #print("TotalBatch=",total)



        return batch



