import argparse
import collections
import json
import logging
import os
import pickle
from multiprocessing import Pool



from generic.misc.config import load_config
import guesswhat.train.utils as utils

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tensorflow.evaluator import Evaluator
from generic.tensorflow.optimizer import create_optimizer

from guesswhat.data_provider.dataset import OracleDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from guesswhat.models.oracle.oracle_network import OracleNetwork



Environment = collections.namedtuple('Environment',  ['trainset', 'validset', 'testset', 'tokenizer'])

###############################
#  LOAD CONFIG
#############################

parser = argparse.ArgumentParser('Oracle network baseline!')

parser.add_argument("-data_dir", type=str, help="Directory with data")
parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
parser.add_argument("-config", type=str, help='Config file')
parser.add_argument("-image_dir", type=str, help='Directory with images')
parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
parser.add_argument("-continue_exp", type=bool, default=False, help="Continue previously started experiment?")
parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

args = parser.parse_args()

config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
logger = logging.getLogger()


###############################
#  LOAD DATA
#############################

(trainset, validset, testset), tokenizer = utils.load_data(
    args.data_dir, load_crop=False, load_picture=False, image_dir=args.image_dir)

trainset = OracleDataset(trainset)
validset = OracleDataset(validset)
testset = OracleDataset(testset)


logger.info('Building network..')
oracle = OracleNetwork(config, len(tokenizer.word2i))


logger.info('Building optimizer..')
optimizer, outputs = create_optimizer(oracle, oracle.loss, config)


###############################
#  START  TRAINING
#############################

# Load config
batch_size = config['optimizer']['batch_size']
no_epoch = config["optimizer"]["no_epoch"]

# create a saver to store/load checkpoint
saver = tf.train.Saver()

#CPU/GPU option
cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

    sources = oracle.get_sources(sess)
    logger.info("Sources: " + ', '.join(sources))

    sess.run(tf.global_variables_initializer())
    start_epoch = utils.load_checkpoint(sess, saver, args, save_path)

    best_val_err = 1e5
    best_train_err = None

    # create training tools
    evaluator = Evaluator(sources, oracle.scope_name)
    oracle_batchifier = OracleBatchifier(tokenizer, sources, **config['model']['crop'])

    for t in range(start_epoch, no_epoch):
        logger.info('Epoch {}..'.format(t + 1))

        train_iterator = Iterator(trainset,
                                  batch_size=batch_size, pool=cpu_pool,
                                  batchifier=oracle_batchifier,
                                  shuffle=True)
        train_loss, train_error = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer])

        valid_iterator = Iterator(validset, pool=cpu_pool,
                                  batch_size=batch_size,
                                  batchifier=oracle_batchifier,
                                  shuffle=False)
        valid_loss, valid_error = evaluator.process(sess, valid_iterator, outputs=outputs)

        logger.info("Training loss: {}".format(train_loss))
        logger.info("Training error: {}".format(train_error))
        logger.info("Validation loss: {}".format(valid_loss))
        logger.info("Validation error: {}".format(valid_error))

        if valid_error < best_val_err:
            best_train_err = train_error
            best_val_err = valid_error
            saver.save(sess, save_path.format('params.ckpt'))
            logger.info("Oracle checkpoint saved...")

        pickle.dump({'epoch': t}, open(save_path.format('status.pkl'), 'wb'))

    saver.restore(sess, save_path.format('params.ckpt'))

    test_iterator = Iterator(testset, pool=cpu_pool,
                             batch_size=batch_size,
                             batchifier=oracle_batchifier,
                             shuffle=True)
    [test_loss, test_error] = evaluator.process(sess, test_iterator, outputs)


    # Experiment done; write results to experiment database (jsonl file)
    with open(os.path.join(args.exp_dir, 'experiments.jsonl'), 'a') as f:
        exp = dict()
        exp['train_error'] = best_train_err
        exp['test_error'] = test_error
        exp['best_val_err'] = best_val_err
        exp['identifier'] = exp_identifier

        f.write(json.dumps(exp))
        f.write('\n')
