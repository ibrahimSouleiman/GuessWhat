"""Create a dictionary file from specified GuessWhat dataset

example
-------
python src/guesswhat/preprocess_data/create_dictionary.py -data_dir=/path/to/guesswhat
"""
import argparse
import collections
import io
import json
import os
import numpy as np
# from gensim.models import word2vec,FastText,KeyedVectors
from pyfasttext import FastText 
from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating dictionary..')


    parser.add_argument("-data_dir", type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-texteType", default="Question" ,type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=3,
                        help='Minimum number of occurences to add word to dictionary')


    args = parser.parse_args()

    
    all_question = []
    word2occ = collections.defaultdict(int)

    tknzr = TweetTokenizer(preserve_case=False)
    
    # model = FastText()
    # model.load_model("data/Embedding/wiki-simple.vec")

    # print(model.nearest_neighbors('teacher'))
    # exit()
    print("Processing train/valid dataset...")
    
    trainset = OracleDataset.load("data", "train")
    validset = OracleDataset.load("data", "valid")

    test = []
    print("array append ...")

    for game in trainset.games:
        question = game.questions[0]
        tokens = tknzr.tokenize(question)
        all_question.append(tokens)
        np.append(all_question, all_question)   

    for game in validset.games:
        question = game.questions[0]
        tokens = tknzr.tokenize(question)
        np.append(all_question, all_question)   



    all_question = np.asarray(all_question)

    print("array saved...")

    np.save("data/list_allquestion1.npy",all_question)

    

    
    


    
