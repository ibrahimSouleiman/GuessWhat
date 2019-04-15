import argparse
import logging
import os
from multiprocessing import Pool
from distutils.util import strtobool

import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump

from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import Embeddings
from generic.data_provider.nlp_utils import GloveEmbeddings

from guesswhat.data_provider.guesswhat_dataset import OracleDataset
from guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.oracle.oracle_network import OracleNetwork
import time

if __name__ == '__main__':

    #############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('Oracle network baseline!')
    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file_question", type=str, default="dict.json", help="Dictionary file name")# default dict_pos_tag
    parser.add_argument("-dict_file_description", type=str, default="dict_Description.json", help="Dictionary file name")
    parser.add_argument("-all_dictfile", type=str, default="data/list_allquestion1.npy", help="Dictionary file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-crop_dir", type=str, help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-continue_exp", type=lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")
    parser.add_argument("-gpu_ratio", type=float, default=0.50, help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=4, help="No thread to load batch")


    args = parser.parse_args()
 
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    # print(save_path)
    # exit()


    logger = logging.getLogger()

    # Load config
    resnet_version = config['model']["image"].get('resnet_version', 50)
    finetune = config["model"]["image"].get('finetune', list())
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]
    use_glove = config["model"]["glove"]




    ###############################
    #  LOAD DATA
    #############################

    # Load image
    image_builder, crop_builder = None, None
    use_resnet = False
    if config['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    if config['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)
        use_resnet = crop_builder.is_raw_image()

    # Load data
    logger.info('Loading data..')

    all_category = {}

    trainset = OracleDataset.load(args.data_dir, "train", all_category,image_builder, crop_builder)
    validset = OracleDataset.load(args.data_dir, "valid", all_category,  image_builder, crop_builder)
    testset = OracleDataset.load(args.data_dir, "test", all_category, image_builder, crop_builder)

    # Load dictionary
    logger.info('Loading dictionary Question..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file_question))

    # Load dictionary
    tokenizer_description = None
    if config["inputs"]["description"]:
        logger.info('Loading dictionary Description......')
        tokenizer_description = GWTokenizer(os.path.join(args.data_dir,args.dict_file_description),question=False)


    # Build Network
    logger.info('Building network..')
    
    if tokenizer_description != None:
        network = OracleNetwork(config, num_words_question=tokenizer.no_words,num_words_description=tokenizer_description.no_words)
    else:        network = OracleNetwork(config, num_words_question=tokenizer.no_words,num_words_description=None)



    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config, finetune=finetune)
    best_param = network.get_predict()
    ###############################
    #  START  TRAINING
    #############################

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()
    resnet_saver = None



    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    # Retrieve only resnet variabes
    if use_resnet:
        resnet_saver = create_resnet_saver([network])
    # use_embedding = False
    # if config["embedding"] != "None":
    #      use_embedding = True
    glove = None
    if use_glove:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(os.path.join(args.data_dir, config["glove_name"]))

    # embedding = None
    # if use_embedding:
    #     logger.info('Loading embedding..')
    #     embedding = Embeddings(args.all_dictfile,total_words=tokenizer.no_words,train=trainset,valid=validset,test=testset,dictionary_file_question=os.path.join(args.data_dir, args.dict_file_question),dictionary_file_description=os.path.join(args.data_dir, args.dict_file_description),description=config["inputs"]["description"],lemme=config["lemme"],pos=config["pos"])


    # CPU/GPU option

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:


        sources = network.get_sources(sess)
        
        out_net = network.get_parameters()[-1]
   
        # logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        if use_resnet:
            resnet_saver.restore(sess, os.path.join(args.data_dir, 'resnet_v1_{}.ckpt'.format(resnet_version)))

        start_epoch = load_checkpoint(sess, saver, args, save_path)

        # best_val_err = 0
        # best_train_err = None

        # # create training tools
        evaluator = Evaluator(sources, network.scope_name,network=network,tokenizer=tokenizer)

        # #train_evaluator = MultiGPUEvaluator(sources, scope_names, networks=networks, tokenizer=tokenizer)
        # #train_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)
        # #eval_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)
        batchifier =  OracleBatchifier(tokenizer, sources, status=config['status'],glove=glove,tokenizer_description=tokenizer_description,args = args,config=config)

        
        # for t in range(start_epoch, no_epoch):
        #     logger.info('Epoch {}..'.format(t + 1))
        #     # print('Epoch {}..'.format(t + 1))

        #     # print(" train_oracle | Iterator ...")
            
        #     t1 = time.time()
        #     train_iterator = Iterator(trainset,
        #                               batch_size=batch_size, pool=cpu_pool,
        #                               batchifier=batchifier,
        #                               shuffle=True)
        #     t2 = time.time()

        #     print(" train_oracle | Iterator  trainset...Total=",t2-t1)

        #     t1 = time.time()

        #     train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer],out_net=best_param)
        #     t2 = time.time()


        #     print(" train_oracle | Iterevaluatorator...Total=",t2-t1)
        #     t1 = time.time()
        #     valid_iterator = Iterator(validset, pool=cpu_pool,
        #                               batch_size=batch_size*2,
        #                               batchifier=batchifier,
        #                               shuffle=False)
        #     t2 = time.time()

        #     print(" train_oracle | Iterator validset...Total=",t2-t1)


        #     t1 = time.time()
        #     valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs)
        #     t2 = time.time()

        #     print(" train_oracle | evaluator ...Total=",t2-t1)


            
        #     # print("Training loss: {}".format(train_loss))
        #     # print("Training error: {}".format(1-train_accuracy))
        #     # print("Validation loss: {}".format(valid_loss))
        #     # print("Validation error: {}".format(1-valid_accuracy))

        #     logger.info("Training loss: {}".format(train_loss))
        #     logger.info("Training error: {}".format(1-train_accuracy))
        #     logger.info("Validation loss: {}".format(valid_loss))
        #     logger.info("Validation error: {}".format(1-valid_accuracy))
            
        #     t1 = time.time()

        #     if valid_accuracy > best_val_err:
        #         best_train_err = train_accuracy
        #         best_val_err = valid_accuracy
        #         saver.save(sess, save_path.format('params.ckpt'))
        #         logger.info("Oracle checkpoint saved...")

        #         pickle_dump({'epoch': t}, save_path.format('status.pkl'))


        #     t2 = time.time()
        #     print(" train_oracle | Condition ...Total=",t2-t1)

        # Load early stopping

        t1 = time.time()
        save_path = "out/oracle/3a29ed734c5d13860f61ccea17f6f90b/{}"
        saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=True)

        print("Output = {}".format(outputs[1]))
        print("Best_param = {}".format(best_param))
        [test_loss, test_accuracy] = evaluator.process(sess, test_iterator,  outputs=outputs + [optimizer],out_net=best_param,inference=True)
        t2 = time.time()
        
        

        print(" train_oracle | Iterator testset  ...Total=",t2-t1)
        # print("Testing loss: {}".format(test_loss))
        # print("Testing error: {}".format(1-test_accuracy))

        logger.info("Testing loss: {}".format(test_loss))
        logger.info("Testing error: {}".format(1-test_accuracy))

