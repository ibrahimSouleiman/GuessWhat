import argparse
import logging
import io
import os

from nltk.tokenize import TweetTokenizer
from generic.utils.file_handlers import pickle_dump
from generic.utils.config import load_config
from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from generic.data_provider.image_loader import get_img_builder

# from vqa.data_provider.vqa_dataset import VQADataset


# wget http://nlp.stanford.edu/data/glove.42B.300d.zip

if __name__ == '__main__':



    parser = argparse.ArgumentParser('Creating GLOVE dictionary.. Please first download http://nlp.stanford.edu/data/glove.42B.300d.zip')

    parser.add_argument("-data_dir", type=str, default="." , help="Path to VQA dataset")
    parser.add_argument("-glove_in", type=str, default="glove.42B.300d.zip", help="Name of the stanford glove file")
    parser.add_argument("-glove_out", type=str, default="glove_dict.pkl", help="Name of the output glove file")
    parser.add_argument("-year", type=int, default=2014, help="VQA dataset year (2014/2017)")
    parser.add_argument("-config", type=str,default="config/oracle/config.json",help='Config file')
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-img_dir", type=str, help='Directory with images')

    args = parser.parse_args()

    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    logger = logging.getLogger()


    print("Loading dataset...")
    # trainset = VQADataset(args.data_dir, year=args.year, which_set="train")
    # validset = VQADataset(args.data_dir, year=args.year, which_set="val")
    # testdevset = VQADataset(args.data_dir, year=args.year, which_set="test-dev")
    # testset = VQADataset(args.data_dir, year=args.year, which_set="test")

    # Load image
    image_builder, crop_builder = None, None
   

    print("load ... train")

    trainset = OracleDataset.load(args.data_dir, "train")
    
    print("load ... valid")
    validset = OracleDataset.load(args.data_dir, "valid")
    
    print("load ... test")
    testset = OracleDataset.load(args.data_dir, "test")


    tokenizer = TweetTokenizer(preserve_case=False)

    print("Loading glove...")
    with io.open(args.glove_in, 'r', encoding="utf-8") as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            print(len(vals))
            exit()
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            
    print("Mapping glove...")
    pickle_dump(vectors, "vectors.pkl")
    exit()

    glove_dict = {}
    not_in_dict = {}
    for _set in [trainset, validset, testset]:
        for g in _set.games:
            words = tokenizer.tokenize(g.question)
            for w in words:
                w = w.lower()
                w = w.replace("'s", "")
                if w in vectors:
                    glove_dict[w] = vectors[w]
                else:
                    not_in_dict[w] = 1

    print("Number of glove: {}".format(len(glove_dict)))
    print("Number of words with no glove: {}".format(len(not_in_dict)))

    for k in not_in_dict.keys():
        print(k)

    print("Dumping file...")
    pickle_dump(glove_dict, args.glove_out)

print("Done!")