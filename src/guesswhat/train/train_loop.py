import argparse
import json
import os
from multiprocessing import Pool
import logging

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer

from guesswhat.models.oracle.oracle_network import OracleNetwork
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM
from guesswhat.models.guesser.guesser_network import GuesserNetwork
from guesswhat.models.looper.basic_looper import BasicLooper

from guesswhat.data_provider.guesswhat_dataset import Dataset

from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.looper_batchifier import LooperBatchifier

from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from generic.utils.config import load_config, get_config_from_xp

from guesswhat.train.utils import get_img_loader, load_checkpoint, test_model

import guesswhat.data_provider as provider


# ------------------------------------------------------
# Initial Testing success ratio: 0.397512335526
# Initial Valid success ratio: 0.410310444079
# Initial Valid (explore) success ratio: 0.383782679739
# ------------------------------------------------------





def compute_stats(sess, batch_size, env, eval_looper, beam_looper, suffix, do_beam=False):
    import numpy as np

    test_score_sampling = []
    no_loop = 1
    for _ in range(no_loop):
        test_score_sampling += [eval_looper.eval(sess, greedy=False, store_games=True,
                                                 iterator=provider.GameIterator(
                                                     env.testset,
                                                     env.tokenizer,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     status=('success', 'failure'),
                                                     pad_to_batch_size=True
                                                 ))]
    provider.dump_samples_into_dataset(eval_looper.looper.storage, save_path, env.tokenizer, name=suffix + ".sampling")
    # save_plots(save_path.format(""), save_path.format(""), suffix+".sampling")

    test_score_greedy = eval_looper.eval(sess, greedy=True,
                                         iterator=provider.GameIterator(
                                             env.testset,
                                             env.tokenizer,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             status=('success', 'failure'),
                                             pad_to_batch_size=True
                                         ))

    logger.info("------------------------------------------------------")
    logger.info("Testing (sampling) success ratio: {} +/- {}".format(np.mean(test_score_sampling), np.std(test_score_sampling)))
    logger.info(test_score_sampling)
    logger.info("Testing (greedy) Valid success ratio: {}".format(test_score_greedy))

    if do_beam:
        test_score_beamsearch = beam_looper.eval(sess, store_games=True,
                                                 iterator=provider.GameIterator(
                                                     env.testset,
                                                     env.tokenizer,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     status=('success', 'failure'),
                                                     pad_to_batch_size=True
                                                 ))
        provider.dump_samples_into_dataset(beam_looper.storage, save_path, env.tokenizer, name=suffix + ".beam")
        # save_plots(save_path.format(""), save_path.format(""), suffix + ".beam")
        logger.info("Testing (BS) success ratio: {}".format(test_score_beamsearch))

    explore_sampling = []
    explore_greedy = []
    for _ in range(no_loop):
        explore_sampling += [eval_looper.eval(sess, provider.LoopIterator(env.trainset, batch_size=batch_size))]
        explore_greedy += [eval_looper.eval(sess, greedy=True, iterator=provider.LoopIterator(env.trainset, batch_size=batch_size))]

    logger.info("------------------------------------------------------")
    logger.info("Explore (sampling) success ratio: {} +/- {}".format(np.mean(explore_sampling), np.std(explore_sampling)))
    logger.info("Explore (greedy) Valid success ratio: {} +/- {}".format(np.mean(explore_greedy), np.std(explore_greedy)))
    logger.info(test_score_sampling)
    logger.info(explore_greedy)

    # if do_beam:
    #     explore_beam_search = []
    #     for _ in range(1):#range(int(no_loop/2+1)):
    #         explore_beam_search += [beam_looper.eval(sess, iterator=provider.LoopIterator(env.trainset, batch_size=1))]
    #     logger.info("Explore (BS) success ratio: {} +/- {} ".format(np.mean(explore_beam_search), np.std(explore_beam_search)))
    #
    #     logger.info(explore_beam_search)
    logger.info("------------------------------------------------------")


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Question generator (policy gradient baseline))')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-image_dir", type=str, help='Directory with images')
    parser.add_argument("-config", type=str, help='Config file')

    parser.add_argument("-networks_dir", type=str, help="Directory with pretrained networks")
    parser.add_argument("-oracle_identifier", type=str, default='-4252052547696363119', help='Oracle identifier')  # Use checkpoint id instead?
    parser.add_argument("-qgen_identifier", type=str, default='6916731124307064098', help='Qgen identifier')
    parser.add_argument("-guesser_identifier", type=str, default='-8261780951461062027', help='Guesser identifier')

    # parser.add_argument("-from_checkpoint", type=bool, default=False, help="Start from checkpoint?")
    parser.add_argument("-from_checkpoint", type=str, help="Start from checkpoint?")

    parser.add_argument("-gpu_ratio", type=float, default=0.95, help="How muany GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")

    args = parser.parse_args()

    loop_config, exp_identifier, save_path = load_config(args.config, args.exp_dir)

    # Load all  networks configs
    oracle_config = get_config_from_xp(os.path.join(args.networks_dir, "oracle"), args.oracle_identifier)
    guesser_config = get_config_from_xp(os.path.join(args.networks_dir, "guesser"), args.guesser_identifier)
    qgen_config = get_config_from_xp(os.path.join(args.networks_dir, "qgen"), args.qgen_identifier)

    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################

    # Load image
    logger.info('Loading images..')
    image_loader = get_img_loader(qgen_config['model']['image'], args.image_dir)
    crop_loader = None  # get_img_loader(guesser_config['model']['crop'], args.image_dir)

    # Load data
    logger.info('Loading data..')
    trainset = Dataset(args.data_dir, "train", image_loader, crop_loader)
    validset = Dataset(args.data_dir, "valid", image_loader, crop_loader)
    testset = Dataset(args.data_dir, "test", image_loader, crop_loader)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, 'dict.json'))

    ###############################
    #  LOAD NETWORKS
    #############################

    logger.info('Building networks..')

    qgen_network = QGenNetworkLSTM(qgen_config["model"], num_words=tokenizer.no_words, policy_gradient=True)
    qgen_var = [v for v in tf.global_variables() if "qgen" in v.name and 'rl_baseline' not in v.name]
    qgen_saver = tf.train.Saver(var_list=qgen_var)

    oracle_network = OracleNetwork(oracle_config, num_words=tokenizer.no_words)
    oracle_var = [v for v in tf.global_variables() if "oracle" in v.name]
    oracle_saver = tf.train.Saver(var_list=oracle_var)

    guesser_network = GuesserNetwork(guesser_config["model"], num_words=tokenizer.no_words)
    guesser_var = [v for v in tf.global_variables() if "guesser" in v.name]
    guesser_saver = tf.train.Saver(var_list=guesser_var)

    loop_saver = tf.train.Saver(allow_empty=False)

    ###############################
    #  REINFORCE OPTIMIZER
    #############################

    logger.info('Building optimizer..')

    pg_variables = [v for v in tf.trainable_variables() if "qgen" in v.name and 'rl_baseline' not in v.name]
    baseline_variables = [v for v in tf.trainable_variables() if "qgen" in v.name and 'rl_baseline' in v.name]

    pg_optimize = create_optimizer(qgen_network, qgen_network.policy_gradient_loss, loop_config,
                                   var_list=pg_variables,
                                   optim=tf.train.GradientDescentOptimizer)
    baseline_optimize = create_optimizer(qgen_network, qgen_network.baseline_loss, loop_config,
                                         var_list=baseline_variables,
                                         optim=tf.train.GradientDescentOptimizer,
                                         apply_update_ops=False)

    optimizer = [pg_optimize, baseline_optimize]

    ###############################
    #  START TRAINING
    #############################

    # Load config
    batch_size = loop_config['optimizer']['batch_size']
    no_epoch = loop_config["optimizer"]["no_epoch"]

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        ###############################
        #  LOAD PRE-TRAINED NETWORK
        #############################

        sess.run(tf.global_variables_initializer())
        if args.from_checkpoint:
            # start_epoch = load_checkpoint(sess, saver, args, save_path)
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(args.networks_dir, 'loop', args.from_checkpoint, 'params.ckpt'))
        else:
            qgen_saver.restore(sess, os.path.join(args.networks_dir, 'qgen', args.qgen_identifier, 'params.ckpt'))
            oracle_saver.restore(sess, os.path.join(args.networks_dir, 'oracle', args.oracle_identifier, 'params.ckpt'))
            guesser_saver.restore(sess, os.path.join(args.networks_dir, 'guesser', args.guesser_identifier, 'params.ckpt'))

        # check that models are correctly loaded
        test_model(sess, testset, cpu_pool=cpu_pool, tokenizer=tokenizer,
                   oracle=oracle_network,
                   guesser=guesser_network,
                   qgen=qgen_network,
                   batch_size=100,
                   logger=logger)

        # create training tools
        loop_sources = qgen_network.get_sources(sess)
        logger.info("Sources: " + ', '.join(loop_sources))

        evaluator = Evaluator(loop_sources, qgen_network.scope_name, network=qgen_network, tokenizer=tokenizer)

        train_batchifier = LooperBatchifier(tokenizer, loop_sources)
        eval_batchifier = QuestionerBatchifier(tokenizer, loop_sources)

        # Initialize the looper to eval/train the game-simulation
        qgen_network.build_sampling_graph(qgen_config["model"], tokenizer=tokenizer, max_length=loop_config['loop']['max_depth'])
        looper_evaluator = BasicLooper(loop_config,
                                       oracle=oracle_network,
                                       guesser=guesser_network,
                                       qgen=qgen_network,
                                       tokenizer=tokenizer)

        # Evaluate starting point
        # logger.info(">>>-------------- INITIALISATION ---------------------<<<")
        # compute_stats(sess, batch_size, env, eval_looper, beam_looper, suffix="start", do_beam=True)
        # logger.info(">>>---------------------------------------------------<<<")

        logs = []
        # Start training
        final_val_score = 0.
        for epoch in range(no_epoch):

            logger.info("Epoch {}/{}".format(epoch, no_epoch))

            train_iterator = Iterator(trainset, batch_size=batch_size,
                                      pool=cpu_pool,
                                      batchifier=train_batchifier,
                                      use_padding=True)
            train_score = looper_evaluator.process(sess, train_iterator,
                                                   optimizer=optimizer)

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size * 2,
                                      batchifier=eval_batchifier,
                                      shuffle=False,
                                      use_padding=True)
            val_score = looper_evaluator.process(sess, valid_iterator)

            logger.info("Train (Explore) success ratio : {}".format(train_score))
            logger.info("Val success ratio : {}".format(val_score))

            if val_score > final_val_score:
                logger.info("Save checkpoint...")
                final_val_score = val_score
                loop_saver.save(sess, save_path.format('params.ckpt'))

        # Compute the test score with early stopping
        loop_saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size * 2,
                                 batchifier=eval_batchifier,
                                 shuffle=False,
                                 use_padding=True)
        test_score = looper_evaluator.process(sess, test_iterator)

        # logger.info(">>>-------------- FINAL SCORE ---------------------<<<")
        # compute_stats(sess, batch_size, env, eval_looper, beam_looper, suffix="model", do_beam=False)
        # logger.info(">>>------------------------------------------------<<<")