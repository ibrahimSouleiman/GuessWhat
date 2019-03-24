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
import nltk
from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer

from nltk import WordNetLemmatizer



def get_lemme(lemmatizer,word):

    lemme = lemmatizer.lemmatize(word)

    return lemme
    
def get_pos(word):
    pos = nltk.pos_tag([word])
    return pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating dictionary..')


    parser.add_argument("-data_dir", type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-texteType", default="Question" ,type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Name of the dictionary file")
    parser.add_argument("-min_occ", type=int, default=0,
                        help='Minimum number of occurences to add word to dictionary')



    args = parser.parse_args()
    lemmas = WordNetLemmatizer()
    

    word2occ = collections.defaultdict(int)

    tknzr = TweetTokenizer(preserve_case=False)


    print("Processing train dataset...")
    trainset = OracleDataset.load(args.data_dir, "train")

    word2i = {'<padding>': [0,get_lemme(lemmas,"padding"),get_pos("padding")],
                '<start>': [1,get_lemme(lemmas,"start"),get_pos("start")],
                '<stop>': [2,get_lemme(lemmas,"stop"),get_pos("stop")],
                '<stop_dialogue>': [3,get_lemme(lemmas,"stop_dialogue"),get_pos("stop_dialogue")],
                '<unk>': [4,get_lemme(lemmas,"unk"),get_pos("unk")],
                '<yes>': [5,get_lemme(lemmas,"yes"),get_pos("yes")],
                '<no>': [6,get_lemme(lemmas,"no"),get_pos("no")],
                '<n/a>': [7,get_lemme(lemmas,"n/a"),get_pos("n/a")],
                }

    

    if args.texteType == "Question":
        # Set default values
        for game in trainset.games:
            question = game.questions[0]
            tokens = tknzr.tokenize(question)
            
            for tok in tokens:
                word2occ[tok] += 1
            

    elif args.texteType == "Description":
        # Set default values
        for game in trainset.games:
            description = game.image.description
            tokens = tknzr.tokenize(description)
            for tok in tokens:
                word2occ[tok] += 1

            
    
    print("filter words...")
    for word, occ in word2occ.items():
        
        if occ >= args.min_occ and word.count('.') <= 1:
            word2i[word] = [len(word2i),get_lemme(lemmas,word),get_pos(word)]



    print("Number of words (occ >= 1): {}".format(len(word2occ)))
    print("Number of words (occ >= {}): {}".format(args.min_occ, len(word2i)))

    dict_path = os.path.join(args.data_dir, args.dict_file)
    print("Dump file: {} ...".format(dict_path))
    with io.open(dict_path, 'wb') as f_out:
        data = json.dumps({'word2i': word2i}, ensure_ascii=False)
        f_out.write(data.encode('utf8', 'replace'))

    print("Done!")
    # print(word2occ["image"])
    # print(word2i["image"])
