"""Create a dictionary file from specified GuessWhat dataset

example
-------
python src/guesswhat/preprocess_data/create_dictionary.py -data_dir=data/
"""
import argparse
import collections
import io
import json
import os
import nltk
import numpy as np
from gensim.models import word2vec,FastText,KeyedVectors

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from nltk.tokenize import TweetTokenizer

from nltk import WordNetLemmatizer
from pathlib import Path
from gensim.test.utils import get_tmpfile


def get_lemme(lemmatizer,word):

    lemme = lemmatizer.lemmatize(word)

    return lemme
    
def get_pos(word):
    pos = nltk.pos_tag([word])
    return pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Creating dictionary..')


    parser.add_argument("-data_dir", type=str,default="data", help="Path where are the Guesswhat dataset")
    parser.add_argument("-type", default="fasttext" ,type=str, help="Path where are the Guesswhat dataset")
    parser.add_argument("-training", default=False ,type=bool, help="Path where are the Guesswhat dataset")

    # parser.add_argument("-dict_file_worddes", type=str, default="dict.json", help="Name of the dictionary file")
    # parser.add_argument("-dict_file_posdes", type=str, default="dict.json", help="Name of the dictionary file")
    # parser.add_argument("-dict_file_word", type=str, default="dict.json", help="Name of the dictionary file")
    # parser.add_argument("-dict_file_pos", type=str, default="dict.json", help="Name of the dictionary file")

    parser.add_argument("-dict_file_question", type=str, default="dict_pos_tag.json", help="Name of the dictionary file")
    parser.add_argument("-dict_file_description", type=str, default="dict_Description.json", help="Name of the dictionary file")



    parser.add_argument("-file_allquestion", type=str, default="all_question", help="Name of the dictionary file")
    parser.add_argument("-file_alldescription", type=str, default="all_description", help="Name of the dictionary file")


    parser.add_argument("-file_lemme_ques", type=str, default="all_lemme_ques", help="Name of the dictionary file")
    parser.add_argument("-file_lemme_des", type=str, default="all_lemme_des", help="Name of the dictionary file")


    parser.add_argument("-file_pos_ques", type=str, default="all_pos_ques", help="Name of the dictionary file")
    parser.add_argument("-file_pos_des", type=str, default="file_pos_des", help="Name of the dictionary file")

    parser.add_argument("-emb_dim", type=int, default=100,help="Name of the dictionary file")


    parser.add_argument("-min_occ", type=int, default=0,
                        help='Minimum number of occurences to add word to dictionary')

    #model_wordd=None,model_posd=None,model_word=None,model_pos=None


    args = parser.parse_args()
    lemmas = WordNetLemmatizer()
    

    word2occ = collections.defaultdict(int)



    print("Processing train dataset...")
    trainset = OracleDataset.load(args.data_dir, "train")
    print("Processing valid dataset...")

    validset = OracleDataset.load(args.data_dir, "valid")


    tknzr = TweetTokenizer(preserve_case=False)
    unk = "<unk>"
    all_questions = [[unk,unk,unk,unk]]
    all_descriptions = [[unk,unk,unk,unk]]


    all_lemmes_ques = [[unk,unk,unk,unk]]
    all_postags_ques = [[unk,unk,unk,unk]]

    all_lemmes_des = [[unk,unk,unk,unk]]
    all_postags_des = [[unk,unk,unk,unk]]



    file_allquestion = Path(os.path.join(args.data_dir,args.file_allquestion))
    file_allLemme_ques = Path(os.path.join(args.data_dir,args.file_lemme_ques))
    file_allpos_ques = Path(os.path.join(args.data_dir,args.file_pos_ques))

    file_alldescription = Path(os.path.join(args.data_dir,args.file_allquestion))
    file_allLemme_des = Path(os.path.join(args.data_dir,args.file_lemme_ques))
    file_allpos_des = Path(os.path.join(args.data_dir,args.file_pos_ques))

    if file_allquestion.is_file() & file_allLemme_ques.is_file() & file_allpos_ques.is_file() & file_alldescription.is_file()  & file_allLemme_des.is_file()  & file_allpos_des.is_file() :
            
            print(" Load List ..")
            all_questions=np.load(file_allquestion)
            all_descriptions=np.load(file_allquestion)
            
            all_lemmes_ques=np.load(file_allLemme_ques)
            all_postags_quess=np.load(file_allpos_ques)

            all_lemmes_des=np.load(file_allLemme_des)
            all_postags_des=np.load(file_allpos_des)
    
    else:
        print(" Create List ..........")
        with open(os.path.join(args.data_dir,args.dict_file_description), 'r') as f:
                word2i_des = json.load(f)['word2i']
        with open(os.path.join(args.data_dir,args.dict_file_description), 'r') as f:
                word2i_question = json.load(f)['word2i']
        
        for game in trainset.get_data():
            data_question = game.questions[0]
            data_description = game.image.description
            
            tokens_question = tknzr.tokenize(data_question)
            tokens_description = tknzr.tokenize(data_description)
            
            # all_lemme = [self.word2i[token][1] for token in tokens]
            all_lemme_ques = []
            all_pos_ques = []
            all_lemme_des = []
            all_pos_des = []
            # nb_erreur_lemme = 0
            # nb_erreur_pos = 0

            for token in tokens_question:
                try:
                    lemme= word2i_question[token][1]
                except KeyError:
                    # nb_erreur_lemme += 1
                    lemme = word2i_question[unk][1]
                
                all_lemme_ques.append(lemme)
                
                try:
                    pos= word2i_question[token][2][0][1] 
                except KeyError:
                    # nb_erreur_pos += 1
                    pos = word2i_question[unk][2][0][1] 
                
                all_pos_ques.append(pos)

            for token in tokens_description:
                try:
                    lemme= word2i_des[token][1]
                except KeyError:
                    # nb_erreur_lemme += 1
                    lemme = word2i_des[unk][1]
                
                all_lemme_des.append(lemme)
                
                try:
                    pos= word2i_des[token][2][0][1] 
                except KeyError:
                    # nb_erreur_pos += 1
                    pos = word2i_des[unk][2][0][1] 
                
                all_pos_ques.append(pos)

            all_questions.append(tokens_question)
            all_descriptions.append(tokens_description)


            all_lemmes_ques.append(all_lemme_ques)
            all_postags_ques.append(all_pos_ques)

            all_lemmes_des.append(all_lemme_des)
            all_postags_des.append(all_pos_des)


        for game in validset.get_data():
            data_question = game.questions[0]
            data_description = game.image.description

            
            
            tokens_question = tknzr.tokenize(data_question)
            tokens_description = tknzr.tokenize(data_description)
            
            # all_lemme = [self.word2i[token][1] for token in tokens]
            all_lemme_ques = []
            all_pos_ques = []
            all_lemme_des = []
            all_pos_des = []
            # nb_erreur_lemme = 0
            # nb_erreur_pos = 0

            for token in tokens_question:
                try:
                    lemme= word2i_question[token][1]
                except KeyError:
                    # nb_erreur_lemme += 1
                    lemme = word2i_question[unk][1]
                
                all_lemme_ques.append(lemme)
                
                try:
                    pos= word2i_question[token][2][0][1] 
                except KeyError:
                    # nb_erreur_pos += 1
                    pos = word2i_question[unk][2][0][1] 
                
                all_pos_ques.append(pos)

            for token in tokens_description:
                try:
                    lemme= word2i_des[token][1]
                except KeyError:
                    # nb_erreur_lemme += 1
                    lemme = word2i_des[unk][1]
                
                all_lemme_des.append(lemme)
                
                try:
                    pos= word2i_des[token][2][0][1] 
                except KeyError:
                    # nb_erreur_pos += 1
                    pos = word2i_des[unk][2][0][1] 
                
                all_pos_des.append(pos)

            all_questions.append(tokens_question)
            all_descriptions.append(tokens_description)


            all_lemmes_ques.append(all_lemme_ques)
            all_postags_ques.append(all_pos_ques)

            all_lemmes_des.append(all_lemme_des)
            all_postags_des.append(all_pos_des)

    


        np.save(os.path.join(args.data_dir,args.file_allquestion),all_questions)
        np.save(os.path.join(args.data_dir,args.file_alldescription),all_descriptions)

        
        np.save(os.path.join(args.data_dir,args.file_lemme_ques),all_lemmes_ques)
        np.save(os.path.join(args.data_dir,args.file_lemme_des),all_lemmes_des)


        np.save(os.path.join(args.data_dir,args.file_pos_ques),all_postags_ques)
        np.save(os.path.join(args.data_dir,args.file_pos_des),all_postags_des)



        print("Fasttext train ...............................")


        model_word_ques = FastText(size=args.emb_dim, window=3, min_count=3) 
        model_word_ques.build_vocab(sentences=all_questions)
        model_word_ques.train(sentences=all_questions, total_words=len(all_questions), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_word_ques.model")
        model_word_ques.save(fname)
        # print(model_word_ques.wv[unk])


        model_lemme_ques = FastText(size=args.emb_dim, window=3, min_count=3) 
        model_lemme_ques.build_vocab(sentences=all_lemmes_ques)
        model_lemme_ques.train(sentences=all_lemmes_ques, total_words=len(all_lemmes_ques), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_lemme_ques.model")
        model_lemme_ques.save(fname)
        # print(model_lemme_ques.wv[unk])

        model_pos_ques = FastText(size=args.emb_dim, window=3, min_count=3) 
        model_pos_ques.build_vocab(sentences=all_postags_ques)
        model_pos_ques.train(sentences=all_postags_ques, total_words=len(all_postags_ques), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_pos_ques.model")
        model_pos_ques.save(fname)

        # print(model_pos_ques.wv[unk])
        # exit()

        print(" ................. Train description")


        model_word_des = FastText(size=args.emb_dim, window=3, min_count=3) 
        model_word_des.build_vocab(sentences=all_descriptions)
        model_word_des.train(sentences=all_descriptions, total_words=len(all_questions), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_word_des.model")
        model_word_des.save(fname)


        model_lemme_des = FastText(size=args.emb_dim, window=3, min_count=3) 
        model_lemme_des.build_vocab(sentences=all_lemmes_des)
        model_lemme_des.train(sentences=all_lemmes_des, total_words=len(all_lemmes_des), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_lemme_des.model")
        model_lemme_ques.save(fname)



        model_pos_des = FastText(size=args.emb_dim, window=3, min_count=3) 

        model_pos_des.build_vocab(sentences=all_postags_des)
        
        model_pos_des.train(sentences=all_postags_des, total_words=len(all_postags_des), epochs=10)

        fname = os.path.join(args.data_dir,"ftext_pos_des.model")
        model_pos_des.save(fname)


        print("Done !")



        # self.model_pos = None
        # if self.pos:
        #     self.model_pos = FastText(size=self.emb_dim, window=3, min_count=0) 
        #     self.model_pos.build_vocab(sentences=all_postags)
        #     self.model_pos.train(sentences=all_postags, total_words=len(all_postags), epochs=5)
        
        # # print(all_lemmes)
        # # print("..... ",all_lemmes[0])
        # # self.model_word[self.unk]
        
        # return self.model_word,self.model_pos

    # word2i = {'<padding>': [0,get_lemme(lemmas,"padding"),get_pos("padding")],
    #             '<start>': [1,get_lemme(lemmas,"start"),get_pos("start")],
    #             '<stop>': [2,get_lemme(lemmas,"stop"),get_pos("stop")],
    #             '<stop_dialogue>': [3,get_lemme(lemmas,"stop_dialogue"),get_pos("stop_dialogue")],
    #             '<unk>': [4,get_lemme(lemmas,"unk"),get_pos("unk")],
    #             '<yes>': [5,get_lemme(lemmas,"yes"),get_pos("yes")],
    #             '<no>': [6,get_lemme(lemmas,"no"),get_pos("no")],
    #             '<n/a>': [7,get_lemme(lemmas,"n/a"),get_pos("n/a")],
    #             }

    

    # if args.texteType == "Question":
    #     # Set default values
    #     for game in trainset.games:
    #         question = game.questions[0]
    #         tokens = tknzr.tokenize(question)
            
    #         for tok in tokens:
    #             word2occ[tok] += 1
            

    # elif args.texteType == "Description":
    #     # Set default values
    #     for game in trainset.games:
    #         description = game.image.description
    #         tokens = tknzr.tokenize(description)
    #         for tok in tokens:
    #             word2occ[tok] += 1

            
    
    # print("filter words...")
    # for word, occ in word2occ.items():
        
    #     if occ >= args.min_occ and word.count('.') <= 1:
    #         word2i[word] = [len(word2i),get_lemme(lemmas,word),get_pos(word)]



    # print("Number of words (occ >= 1): {}".format(len(word2occ)))
    # print("Number of words (occ >= {}): {}".format(args.min_occ, len(word2i)))

    # dict_path = os.path.join(args.data_dir, args.dict_file)
    # print("Dump file: {} ...".format(dict_path))
    # with io.open(dict_path, 'wb') as f_out:
    #     data = json.dumps({'word2i': word2i}, ensure_ascii=False)
    #     f_out.write(data.encode('utf8', 'replace'))

    # print("Done!")
    # # print(word2occ["image"])
    # # print(word2i["image"])
