import numpy as np
import json
from generic.utils.file_handlers import pickle_loader

from gensim.models import word2vec,FastText,KeyedVectors
# from pyfasttext import FastText

#import glove
from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer

#from guesswhat.data_provider.lemmatize import lemmatize
import os
from pathlib import Path

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer



import time

class GloveEmbeddings(object):

    def __init__(self, file, glove_dim=300):
        self.glove = pickle_loader(file)
        self.glove_dim = glove_dim

    def get_embeddings(self, tokens):
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")
            if token in self.glove:
                vectors.append(np.array(self.glove[token]))
            else:
                vectors.append(np.zeros((self.glove_dim,)))
        return vectors

class Embeddings(object):

    def __init__(self, file,total_words=0,emb_dim=100,emb_window=3,embedding="fasttext",train=None,valid=None,test=None,dictionary_file_question="dict.json",dictionary_file_description="dict_Description.json",lemme=False,pos=False,description=False):

        self.unk = "<unk>"
        self.lemme = lemme
        self.pos = pos
        self.description=description

        self.emb_dim = emb_dim
        self.emb_window = emb_window
        self.embedding = embedding
        self.train = train
        self.valid = valid
        self.test = test
        
            
        # self.dictionary_file_question = dictionary_file_question
        # self.dictionary_file_description = dictionary_file_description
      
        # print(" nlp_utls | start to create list_question [] ...")

        # self.model_word,self.model_pos = self.build("all_question.npy","all_lemmes.npy","all_pos.npy",dictionary_file=self.dictionary_file_question)
        # if self.description:
        #      self.model_wordd,self.model_posd = self.build("all_description.npy","all_dlemme.npy","all_dpos.npy",dictionary_file=self.dictionary_file_description)
        
        
     

    def build(self,file_question,file_lemme,file_pos,dictionary_file):

        # file_allquestion = Path("data/all_question.npy")
        # file_allLemme = Path("data/all_lemmes.npy")
        # file_allpos = Path("data/all_pos.npy")
        tknzr = TweetTokenizer(preserve_case=False)

        all_questions = [[self.unk]]
        all_lemmes = [[self.unk]]
        all_postags = [[self.unk]]

    

        file_allquestion = Path("data/"+file_question)
        file_allLemme = Path("data/"+file_lemme)
        file_allpos = Path("data/"+file_pos)

    
        if file_allquestion.is_file() & file_allLemme.is_file() & file_allpos.is_file():
             all_questions=np.load(file_allquestion)
             all_lemmes=np.load(file_allLemme)
             all_postags=np.load(file_allpos)
       
        else:
            with open(dictionary_file, 'r') as f:
                  self.word2i = json.load(f)['word2i']
            
            for game in self.train.games:
                if self.description:
                    data = game.image.description
                    # print(game.questions[0])
                    # print("--- if=",data)
                    # print("---- else = ",game.questions[0])
                     
                else:
                    data = game.questions[0]
                     

              
                
                tokens = tknzr.tokenize(data)
                
                # all_lemme = [self.word2i[token][1] for token in tokens]
                all_lemme = []
                all_pos = []
                nb_erreur_lemme = 0
                nb_erreur_pos = 0

                for token in tokens:
                    try:
                        lemme= self.word2i[token][1]
                    except KeyError:
                        nb_erreur_lemme += 1
                        lemme = self.word2i[self.unk][1]
                   
                    all_lemme.append(lemme)
                   
                    try:
                        pos= self.word2i[token][2][0][1] 
                    except KeyError:
                        nb_erreur_pos += 1
                        pos = self.word2i[self.unk][2][0][1] 
                    
                    all_pos.append(pos)



                # all_pos = [self.word2i[token][2][0][1] for token in tokens]


                all_questions.append(tokens)
                all_lemmes.append(all_lemme)
                all_postags.append(all_pos)


            # print(" nlp_ulis | finish train .....")
            # for game in self.valid.games:

            #     question = game.questions[0]
            #     tokens = tknzr.tokenize(question)
                
            #     all_lemme = [self.word2i[token][1] for token in tokens]
            #     all_pos = [self.word2i[token][2] for token in tokens]


            #     all_questions.append(tokens)
            #     all_lemmes.append(all_lemme)
            #     all_postags.append(all_lemme)



            # print(" nlp_ulis | finish valid .....")

            # for game in self.test.games:

            #     question = game.questions[0]
            #     tokens = tknzr.tokenize(question)
                
            #     all_lemme = [self.word2i[token][1] for token in tokens]
            #     all_pos = [self.word2i[token][2] for token in tokens]

            #     all_questions.append(tokens)
            #     all_lemmes.append(all_lemme)
            #     all_postags.append(all_lemme)

            print(" nlp_ulis | finish test .....")
            
            np.save("data/"+file_question,all_questions)
            np.save("data/"+file_lemme,all_lemmes)
            np.save("data/"+file_pos,all_postags)


           
        #self.lemmatize = lemmatize()



        self.tknzr = tknzr
        self.model = None




        self.tknzr = tknzr
        self.model_word = None
        self.embedding = self.embedding
        self.total_words = self.total_words


        
        self.data = all_questions

        if self.lemme:
            print("-----------------------****** -- Lemme true nlp_utilis",all_lemmes[1])
            self.data = all_lemmes

        
        
        self.model_word = FastText(size=self.emb_dim, window=3, min_count=0) 
        self.model_word.build_vocab(sentences=self.data)
        self.model_word.train(sentences=self.data, total_words=len(self.data), epochs=5)

        self.model_pos = None
        if self.pos:
            self.model_pos = FastText(size=self.emb_dim, window=3, min_count=0) 
            self.model_pos.build_vocab(sentences=all_postags)
            self.model_pos.train(sentences=all_postags, total_words=len(all_postags), epochs=5)
        
        # print(all_lemmes)
        # print("..... ",all_lemmes[0])
        # self.model_word[self.unk]
        
        return self.model_word,self.model_pos

            
       




def get_embeddings(tokens_word,lemme=False,pos=False,description=False,embedding="fasttext",model_wordd=None,model_worddl=None,model_word=None,model_wordl=None,model_pos=None,model_posd=None):
    """
    tokens_word : all word 

    """
    vectors = []
    


    pos_vectors = []
    unk = "<unk>"


    if embedding == "fasttext":
        
        for token in tokens_word:
            
            if description:

                token = token.replace("'s","")
                try:
                    if lemme:
                        vectors.append(np.asarray(model_worddl[token]))
                    else:vectors.append(np.asarray(model_wordd[token]))

                
                except KeyError:
                    # print("_____________ 1Unknow=",self.unk)
                    vectors.append(np.asarray(model_wordd[unk]))
                
                if pos:
                    try:
                        pos_vectors.append(np.asarray(model_posd[token]))
                    except KeyError:
                        # print("_____________ 2Unknow=",self.unk)
                        pos_vectors.append(np.asarray(model_posd[unk]))
            else:
                token = token.replace("'s","")
                try:
                    if lemme:
                        vectors.append(np.asarray(model_word[token]))
                    else:vectors.append(np.asarray(model_wordl[token]))
                
                except KeyError:
                    # print("_____________ 3Unknow=",self.unk)
                    vectors.append(np.asarray(model_word[unk]))
                
                if pos:
                    try:
                        pos_vectors.append(np.asarray(model_pos[token]))
                    except KeyError:
                        # print("_____________ 4Unknow=",self.unk)
                        pos_vectors.append(np.asarray(model_pos[unk]))


    return vectors,pos_vectors

# def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

#     if seq_length is None:
#         seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)
#         #print("nlp | sequence_length = {} ",seq_length)


#     if max_seq_length == 0:
#         max_seq_length = seq_length.max()

#     batch_size = len(list_of_tokens)
#     # print("-- nlp | list_of_tokens= {}".format(list_of_tokens))
#     # print("--- nlp | batch_size = {} ,max_seq_length={} ",batch_size,max_seq_length)
#     padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)
#     # print("--- nlp | padded_tokens={}")
#     # print(" --- list_of_tokens | = {}",list_of_tokens)
#     # print(" --- list_of_tokens[0] | = {}",list_of_tokens[0])

#     for i, seq in enumerate(list_of_tokens):

#         # print("---- 1 seq=",seq)
#         seq = seq[:max_seq_length]
#         # print("---- 2 seq=",len(seq),len(seq[0]))
#         # print("--- 3.1",padded_tokens[i, :len(seq)][0])
#         # print(" ---- 3 seq = ",len(padded_tokens[i, :len(seq)]),len(padded_tokens[i, :len(seq)][0]))
#         padded_tokens[i, :len(seq)] = seq

#     # print("-- NL_UTILS | Max_Seq = {}".format(max_seq_length))

#     return padded_tokens, seq_length
def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

    if seq_length is None:
        seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)
        #  print("nlp | sequence_length = {} ",seq_length)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)

    padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq)] = seq

    return padded_tokens, seq_length

def padder_3d(list_of_tokens, max_seq_length=0):
    seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)

    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    feature_size = list_of_tokens[0][0].shape[0]

    padded_tokens = np.zeros(shape=(batch_size, max_seq_length, feature_size))

    for i, seq in enumerate(list_of_tokens):
        seq = seq[:max_seq_length]
        padded_tokens[i, :len(seq), :] = seq

    # max_seq_length
    return padded_tokens, seq_length


class DummyTokenizer(object):
    def __init__(self):
        self.padding_token = 0
        self.dummy_list = list()
        self.no_words = 10
        self.no_answers = 10
        self.unknown_answer = 0

    def encode_question(self, _):
        return self.dummy_list

    def encode_answer(self, _):
        return self.dummy_list
