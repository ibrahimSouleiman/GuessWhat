import numpy as np
import json
from generic.utils.file_handlers import pickle_loader

from gensim.models import word2vec,FastText,KeyedVectors
# from pyfasttext import FastText
#import glove
from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer


class Embeddings(object):

    def __init__(self, file,total_words=0,emb_dim=100,embedding="fasttext"):
        print("**************** File=",file)
        self.data = np.load(file)
        self.emb_dim = emb_dim
        self.model = None
        self.embedding = embedding
        self.total_words = total_words

        
        

    def get_embeddings(self, tokens_word):
        vectors = []

        if self.embedding == "fasttext":
            self.model = FastText(size=100, window=3, min_count=1) 
            self.model.build_vocab(sentences=self.data)
            self.model.train(sentences=self.data, total_words=len(self.data), epochs=5)
            # self.model.load_model("data/Embedding/wiki-simple.vec")
            
            # print("nlp_utils | ",tokens_word)

            for token in tokens_word:
                # print(self.model[token])
                try:
                    vectors.append(np.asarray(self.model[token]))
                except KeyError:
                    vectors.append(np.zeros((self.emb_dim,)))

                # print(self.model.wv.most_similar("dog"))

        # for token in tokens_word:
        #     token = token.lower().replace("\'s", "")    
        #     if token in self.tokens.append(token):
        #         vectors.append(np.array(self.model[token]))
        #     else:
        #         vectors.append(np.zeros((self.emb_dim)))


        return vectors

def padder(list_of_tokens, seq_length=None, padding_symbol=0, max_seq_length=0):

    if seq_length is None:
        seq_length = np.array([len(q) for q in list_of_tokens], dtype=np.int32)
        #print("nlp | sequence_length = {} ",seq_length)


    if max_seq_length == 0:
        max_seq_length = seq_length.max()

    batch_size = len(list_of_tokens)
    # print("-- nlp | list_of_tokens= {}".format(list_of_tokens))
    # print("--- nlp | batch_size = {} ,max_seq_length={} ",batch_size,max_seq_length)
    padded_tokens = np.full(shape=(batch_size, max_seq_length), fill_value=padding_symbol)
    # print("--- nlp | padded_tokens={}")
    # print(" --- list_of_tokens | = {}",list_of_tokens)
    # print(" --- list_of_tokens[0] | = {}",list_of_tokens[0])

    for i, seq in enumerate(list_of_tokens):

        # print("---- 1 seq=",seq)
        seq = seq[:max_seq_length]
        # print("---- 2 seq=",len(seq),len(seq[0]))
        # print("--- 3.1",padded_tokens[i, :len(seq)][0])
        # print(" ---- 3 seq = ",len(padded_tokens[i, :len(seq)]),len(padded_tokens[i, :len(seq)][0]))


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

    return padded_tokens, max_seq_length


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
