import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image
from generic.data_provider.nlp_utils import padder,padder_3d,padder_4d

import time

from generic.data_provider.nlp_utils import Embeddings,get_embeddings
from gensim.models import word2vec,FastText,KeyedVectors




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

        if self.config["model"]["fasttext"] : 
        
            self.model_worddl = FastText.load(os.path.join("data","ftext_lemme_des.model"))
            self.model_wordd = FastText.load(os.path.join("data","ftext_word_des.model"))        
            self.model_posd = FastText.load(os.path.join("data","ftext_pos_des.model"))
            self.model_wordl = FastText.load(os.path.join("data","ftext_lemme_ques.model"))
            self.model_word = FastText.load(os.path.join("data","ftext_word_ques.model"))
            self.model_pos = FastText.load(os.path.join("data","ftext_pos_ques.model"))
        
        elif self.config["model"]["glove"]:
            self.glove = glove



    
        # self.embedding = Embeddings(args.all_dictfile,total_words=tokenizer_question.no_words,train=trainset,valid=None,test=None,dictionary_file_question=os.path.join(args.data_dir, args.dict_file_question),dictionary_file_description=os.path.join(args.data_dir, args.dict_file_description),description=config["inputs"]["description"],lemme=config["lemme"],pos=config["pos"])




    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games


    def apply(self, games):
        sources = self.sources
        t1 = time.time()

        t1 = time.time()
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            image = game.image
            question = self.tokenizer_question.apply(game.questions[0])
            #print("question =____",game.questions[0])
            # print("tokenize = ___",self.tokenizer_question.apply(game.questions[0]))
            # print(question)
            # exit()
            batch['question'].append(question)
            # print("---------------- FINISH QUESTION=",question)
        

            if 'embedding_vector_ques' in sources:
                
                assert  len(game.questions) == 1
                # Add glove vectors (NB even <unk> may have a specific glove)
                # print("oracle_batchifier | question = {}".format(game.questions[0]))
                words = self.tokenizer_question.apply(game.questions[0],tokent_int=False)
                # print(" End ................... ,",words)
                # print(len(words[0]))
                # print(words)
                # print("/////////// embedding_vector=")
                if type(words) == int:
                    exit()
                
                if "question_pos" in sources:
                    # print("/////////// question_pos")
                    embedding_vectors,embedding_pos = get_embeddings(words,pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos) # slow (copy gloves in process)
                    # print("..... question_pos............. embedding_vectors",len(embedding_vectors[0]))
                    batch['embedding_vector_ques'].append(embedding_vectors)
                    batch['embedding_vector_ques_pos'].append(embedding_pos)
                    batch['question_pos'].append(question)

                    # print(batch['embedding_vector_pos'])
                else:
                    # print("/////////// question_pos NOT EXIST")

                    if self.config["model"]["fasttext"] : 
                        #print("++++++----- ++++++++ Dans fasttext ")
                        embedding_vectors,_ = get_embeddings(words,pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos)
                    elif self.config["model"]["glove"] : 
                        #print("++++++----- ++++++++ Dans glove ")
                       
                        embedding_vectors = self.glove.get_embeddings(words)
                    
                    # print("************ Embedding_vector = ",embedding_vectors)
                    
                    #print("taille = {} ".format(embedding_vectors))
                    # exit()=
                    # print("////////// embedding_vectors=",len(embedding_vectors[0]))

                    batch['embedding_vector_ques'].append(embedding_vectors)

                # print(" Oracle_batchifier | embedding_vector= {}".format(embedding_vectors))
                # print("---- Embedding = ",len(embedding_vectors))
                # print("----  =",len(embedding_vectors[0]))
                #print("---------------- FINISH QUESTION_Emb =",np.asarray(embedding_vectors).shape)





                # games.question = ['am I a person?'],            
                # batch['question'].append(self.tokenizer_question.apply(game.questions[0]))





            if 'embedding_vector_ques_hist' in sources:
                
                assert  len(game.questions) == 1
                # Add glove vectors (NB even <unk> may have a specific glove)
                # print("oracle_batchifier | question = {}".format(game.questions))
                # print("history=",game.all_last_question)
                words = []
                for i in range(6):
                    question_answer = game.all_last_question[i]

                    if len(question_answer) > 1: 
                        # print("QUESTION=",game.all_last_question[i])
                        word = self.tokenizer_question.apply(game.all_last_question[i][1][0],tokent_int=False)
                        # print(" End ................... ,",word,game.all_last_question[i][0])
                        # word = word.extend("Yes")
                        # print("Word = ",word)
                        words.append(word)
                    else:
                        word = self.tokenizer_question.apply(game.all_last_question[i][0],tokent_int=False)
                        words.append(word)
                    


                
                # print(len(words[0]))
                # print(words)
                # print("/////////// embedding_vector=")
                if type(words) == int:
                    exit()
                
            

                if self.config["model"]["fasttext"] : 
                    #print("++++++----- ++++++++ Dans fasttext ")
                    embedding_vectors = []
                    for i in range(6):
                        embedding_vector,_ = get_embeddings(words[i],pos=self.config["model"]["question"]["pos"],lemme=self.config["model"]["question"]["lemme"],model_wordd=self.model_wordd,model_worddl=self.model_worddl,model_word=self.model_word,model_wordl=self.model_wordl,model_posd=self.model_posd,model_pos=self.model_pos)
                        embedding_vectors.append(embedding_vector)
                
                
                elif self.config["model"]["glove"] : 
                    #print("++++++----- ++++++++ Dans glove ")
                    embedding_vectors = []
                    for i in range(6):
                        # print("q=",words[i])
                        embedding_vector= self.glove.get_embeddings(words[i])
                        embedding_vectors.append(embedding_vector)


                    # embedding_vectors = self.glove.get_embeddings(words)


                 


                batch['embedding_vector_ques_hist'].append(embedding_vectors)

                






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



                    batch['embedding_vector_des'].append(embedding_vectors)


            if 'answer' in sources:
                assert len(game.answers) == 1
                batch['answer'].append(answer_dict[game.answers[0]])
                #print(" Correct Answer = ",game.answers[0])

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'allcategory' in sources:
                allcategory = []
                allcategory_hot = np.zeros(shape=(90),dtype=int)
                # print("Oracle_batchifier |  Allcategory -------------------------------")
                for obj in game.objects:
                    allcategory.append(obj.category_id - 1)



                allcategory_hot[allcategory] = 1
                # print("...   ",allcategory,allcategory_hot)

                batch['allcategory'].append(allcategory_hot)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())
                # batch['image_id'].append(image.get_idimage())
                # batch['crop_id'].append(game.object_id)
                # print("crop_id=",game.object_id)
                
            if 'image' in sources:
                batch['image'].append(image.get_image())

                #print("---------------- FINISH IMAGE=",image.get_image().shape)

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
        
        
        
        if "question" in sources:
            batch['question'] , batch['seq_length_question'] = padder(batch['question'],
                                                        padding_symbol=self.tokenizer_question.padding_token)



        if "question_pos" in sources:
            batch['question_pos'], batch['seq_length_ques_pos'] = padder(batch['question_pos'],
                                                            padding_symbol=self.tokenizer_question.padding_token)

        # if "description" in sources:
        #     batch['description'], batch['seq_length_description'] = padder(batch['description'],
        #                                                     padding_symbol=self.tokenizer_question.padding_token)
        

        # batch['embedding_vector_pos'], _ = padder_3d(batch['embedding_vector_pos'])

        if 'embedding_vector_ques' in sources:
                        # print("Shape=",np.asarray(batch['embedding_vector_ques'] ).shape)
                        batch['embedding_vector_ques'],s = padder_3d(batch['embedding_vector_ques'],type_input="question")
                        # print("+++++ Batch = ",batch['seq_length_question'])
                        # print("Len=",len(batch['seq_length_question']))

        if 'embedding_vector_ques_hist' in sources:
                        # print("Shape=",np.asarray(batch['embedding_vector_ques_hist'] ).shape)
                        batch_hist, size_sentences,max_seq = padder_4d(batch['embedding_vector_ques_hist'])

                        batch_hist = np.asarray(batch_hist)
                        size_sentences = np.asarray(size_sentences)

                        # print("All before = ",batch_hist.shape)

                        # batch_hist = np.reshape(batch_hist,(-1,6*max_seq,300))
                        
                        # print("All after = ",batch_hist.shape)
                        # print("+++++ Batch = ",np.asarray(size_sentences).shape)

                        batch['embedding_vector_ques_hist'] = batch_hist

                        for i in range(6):
                            # print("size = {} , {} ,{}".format(size_sentences.shape , size_sentences[:,i].shape ,batch_hist[:,i,:,:].shape )) 
                            
                            # print(size_sentences[:,i])

                            batch['embedding_vector_ques_hist_H{}'.format(i)] = batch_hist[:,i,:,:]
                            batch['seq_length_question_history_H{}'.format(i)] = size_sentences[:,i]

                        print("Finish .........")

                            

                        #print("Len=",len(batch['seq_length_question']))


        if 'embedding_vector_ques_pos' in sources:
                        batch['embedding_vector_ques_pos'], _ = padder_3d(batch['embedding_vector_ques_pos'])

        if 'embedding_vector_des' in sources:
                        #print("++++++++++++++ EMbedding_vector_Des ",batch['embedding_vector_des'])
                        batch['embedding_vector_des'], batch['seq_length_description'] = padder_3d(batch['embedding_vector_des'])

        if 'embedding_vector_des_pos' in sources:
                        batch['embedding_vector_des_pos'], _ = padder_3d(batch['embedding_vector_des_pos'])




        # if 'description' in sources:
        #     # complete par padding en prenons la taille maximal
        # batch['description'], batch['seq_length_description'] = padder_3d(batch['description'])











        


        # print("finish oracle_bachifier .... time=",total)

        # print("TotalBatch=",total)


        #print("TotalBatch=",total)



        return batch



