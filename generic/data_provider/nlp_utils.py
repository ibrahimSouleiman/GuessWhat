import numpy as np
import json
from random import shuffle
import re
import urllib.request
import zipfile
import lxml.etree
import urllib.request

# from pyfasttext import FastText

#import glove


from nltk.tokenize import TweetTokenizer
from gensim.scripts.glove2word2vec import glove2word2vec
from pathlib import Path
from generic.utils.file_handlers import pickle_loader
from gensim.models import Word2Vec,FastText,KeyedVectors
from pathlib import Path
#from guesswhat.data_provider.lemmatize import lemmatize
import os
import numpy as np
import path



import time

class GloveEmbeddings(object):

    def __init__(self, file, glove_dim=100,input_file = 'glove.6B.100d.txt',type_data="wikipedia" ):

        if glove_dim == 300:
            input_file = 'glove.42B.300d.txt'
        


        self.glove_dim = glove_dim
        self.glove_input_file = os.path.join("data",input_file)
        self.word2vec_output_file = input_file + "{}_.word2vec".format(type_data)

        self.file_word2vec = Path(os.path.join("data",self.word2vec_output_file))
        self.filename = os.path.join("data",self.word2vec_output_file)

        ## Extraction only_word used in guessWhat


        self.glove_Wonly = Path(os.path.join("data","glove_onlyWord_GuessWhat_{}_{}.txt".format(glove_dim,"wikipedia")))

      
    
        if self.glove_Wonly.is_file() == False:
            self.all_word = Path(os.path.join("data","all_word.npy"))
            dict_all_word_values = np.load(self.all_word)
            dict_glove_words = {}

            dict_all_word = dict_all_word_values.item().keys()
            print(len(dict_all_word))
           
            # read all_word and write new_file only word in the guessWhat
            with open(self.glove_input_file,"r") as file_glove:
                with open(self.glove_Wonly,"a") as file_out:
                    for line in file_glove:
                            one_line = line.split(" ")
                            word = one_line[0]
                            embedding = one_line[1:]
                            if word in dict_all_word:
                                file_out.write(line)


        if self.file_word2vec.is_file() == False:
            glove2word2vec(self.glove_Wonly, self.filename)
        self.glove = KeyedVectors.load_word2vec_format(self.filename, binary=False)
       
        self.unk = "<unk>"
  


    def get_embeddings(self, tokens):
        # print("........... get_embeddings")
        vectors = []
        for token in tokens:
            token = token.lower().replace("\'s", "")            
            try:
                emb_token = self.glove[token]
                vectors.append(np.array(self.glove[token]))
            except KeyError:
                vectors.append(np.zeros((self.glove_dim,)))



        return vectors

class Embeddings(object):

    def __init__(self, file_name,total_words=0,emb_dim=100,emb_window=3,embedding_name="fasttext",train=None,valid=None,test=None,dictionary_file_question="dict.json",dictionary_file_description="dict_Description.json",lemme=False,pos=False,description=False):

        self.file_name = file_name

        self.unk = "<unk>"
        self.lemme = lemme
        self.pos = pos
        self.description=description

        self.emb_dim = emb_dim
        self.emb_window = emb_window
        self.embedding_name = embedding_name

        self.train = train
        self.valid = valid
        self.test = test

        name_file = "saved_model_{}_doc_ques".format(embedding_name)
        model_saved = Path(name_file)

        if model_saved.is_file():
            self.model =  FastText.load(name_file)
            print("file already exist ...")
        
        else:
            self.input_text = self.get_data(self.file_name)
            self.get_list_data = self.get_list_data(self.input_text)
            self.model = self.get_model_embedding(self.get_list_data)
            self.model.save('saved_model_fasttext_doc_ques')
            print("file not exist ...")


        



    
    
    def get_data(self,name_files):
        
        urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")

        input_text = ""
        for name_file in name_files:
            extension_file = name_file.split(".")[-1]
            print("name_file = {}".format(name_file))
            if extension_file == "zip":
                with zipfile.ZipFile(name_file, 'r') as z:
                    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
                input_text += '\n'.join(doc.xpath('//content/text()'))

            elif extension_file == "txt":
                with open(name_file,'r') as f:
                    input_text += f.read()

            
        return input_text

    def get_list_data(self,data):

        input_text_noparens = re.sub(r'\([^)]*\)', '', data)
        # store as list of sentences
        sentences_strings_ted = []
        
        for line in input_text_noparens.split('\n'):
            m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
            sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
        
        # store as list of lists of words
        sentences_ted = []
        for sent_str in sentences_strings_ted:
            tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
            sentences_ted.append(tokens)
            
        return sentences_ted 

    def get_model_embedding(self,sentences):

        self.model = None
        if self.embedding_name == "fasttext":
            self.model = FastText(sentences, size=self.emb_dim, window=5, min_count=5, workers=4,sg=1)
            
        elif self.embedding_name == "glove":
            self.model = Word2Vec(sentences=sentences, size=self.emb_dim, window=5, min_count=5, workers=4, sg=0)
    
        return self.model


    def get_embedding(self,words):

        assert isinstance(words, list),"get_embedding have param list only"

        vector_zeros = np.zeros(self.emb_dim)
        self.all_embedding = []
        
        try:
            # print("embedding  = {} ".format(self.model[words]))
            # print("words = {} ".format(words))
            self.all_embedding = self.model[words]
        except Exception:
            print("EXCEPTION **** ")
            for word in words:
                try:
                    # self.all_embedding.append( self.model[word])
                    self.all_embedding.append(vector_zeros)
                except Exception:
                    self.all_embedding.append( vector_zeros ) 

        self.all_embedding = np.asarray(self.all_embedding)

        return self.all_embedding
        




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

    return np.zeros((300)),600

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
                
                except Exception:
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


def padder_4d(list_of_tokens,max_seq_length=0):
    seq_length = np.array([[len(s) for s in q] for q in list_of_tokens], dtype=np.int32)
    
    memory_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)
    memory_length = memory_length.max()

    # seq_length = np.array([ q.max() for q in seq_length    ], dtype=np.int32)
    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    # print("Batch = ",batch_size)

    feature_size = list_of_tokens[0][0][0].shape[0]
    
    padded_tokens = np.zeros(shape=(batch_size,memory_length, max_seq_length, feature_size))
    
    for i, hist in enumerate(list_of_tokens):
        for j,h in enumerate(hist):
            # print("Shape = ",np.asarray(h).shape)
            
            seq = h[:max_seq_length]
            # print("Seq = {}".format(np.asarray(seq).shape ))
            # if j == 5:
            #    exit()
            padded_tokens[i,j, :len(seq), :] = seq

    # max_seq_length

    return padded_tokens, seq_length,max_seq_length


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
