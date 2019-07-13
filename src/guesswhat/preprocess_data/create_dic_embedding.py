from nltk.tokenize import TweetTokenizer
from generic.data_provider.nlp_utils import Embeddings,get_embeddings

import json
import numpy  as np
import pickle
from generic.utils.config import load_json_config



#############################
config = load_json_config("config/oracle/config.json")

tokenizer = TweetTokenizer(preserve_case=True)
type_mebdding  = config["model"]["question"]["embedding_type"]

if config["model"]["question"]["embedding_type"] == 0:
    embedding = Embeddings(file_name=["all_question_game.txt" ],embedding_type = type_mebdding , embedding_name="fasttext",emb_dim=100)
elif config["model"]["question"]["embedding_type"] == 1:
    embedding = Embeddings(file_name=["ted_en-20160408.zip" ],embedding_type = type_mebdding,embedding_name="fasttext",emb_dim=100)
elif config["model"]["question"]["embedding_type"] == 2:
    embedding = Embeddings(file_name=["ted_en-20160408.zip","all_question_game.txt" ],embedding_type = type_mebdding, embedding_name="fasttext",emb_dim=100)

#############################
_allWords = {}
dict_all_words = {}
all_words = []
#############################

with open("all_question_game.txt","r") as f:
    all_question = f.read()
    words = tokenizer.tokenize(all_question.lower())    
    print("words_len = {} ".format(len(words)))


all_words = list(set(words))

# for i,word in enumerate(all_words):
    
with open("data/dict.json", "r") as f:
    word2i = json.load(f)['word2i']



# diff_words = []
# for i,word in enumerate(all_words) :
#     if word not in word2i.keys():
#         diff_words.append(word)

# print("diff = {}".format(diff_words))


# print("taille diff = {} ".format(len(diff_words)))
# print("word2i len = {}".format(len(word2i)))
# print("all_words queston len =  ",len(all_words))
# exit()


# last_id = 12130
dict_all_embedding = np.zeros((len(word2i.keys()),100))

for i,word in enumerate(word2i.keys()):
    dict_all_embedding [word2i[word][0]] = embedding.get_embedding([word])
    if i == 12130:
        print("word = {}".format(word))
        embedding_1 = embedding.get_embedding([word])
        embedding_2 = dict_all_embedding [i]
        print(embedding_1 == embedding_2)
       
print("create pickle file ...")

# with open("data/dict_word_indice.pickle","wb") as f:
#     pickle.dump(dict_all_words,f,pickle.HIGHEST_PROTOCOL)

with open("data/dict_word_embedding_{}_{}.pickle".format("fasttext",type_mebdding),"wb") as f:
    pickle.dump(dict_all_embedding,f,pickle.HIGHEST_PROTOCOL)


print("load pickle file ...")

print(word2i)
with open("data/dict_word_embedding_{}_{}.pickle".format("fasttext",type_mebdding),"rb") as f:
    dict_all_embedding = pickle.load(f)
    

# with open("data/dict_word_indice.pickle","rb") as f:
#     dict_all = pickle.load(f)
#     print(dict_all)

# print("all words  = {}".format(len(dict_all_embedding)))
# print(dict_all_embedding)
# print("finish ...")
embedding_1 = dict_all_embedding[10448]
embedding_2 = embedding.get_embedding(["kissed"])

print(embedding_1)
print(embedding_2)


print(embedding_1 == embedding_2)


# print(len(dict_all_embedding))
# print("dict_allWords",dict_allWords)





    
   
 


























