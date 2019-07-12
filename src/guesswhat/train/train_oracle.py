import argparse
import logging
import os
from multiprocessing import Pool
from distutils.util import strtobool

import tensorflow as tf

import numpy as np
import json
from pathlib import Path

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint, create_resnet_saver
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder
from generic.data_provider.nlp_utils import Embeddings
from generic.data_provider.nlp_utils import GloveEmbeddings

from src.guesswhat.data_provider.guesswhat_dataset import OracleDataset
from src.guesswhat.data_provider.oracle_batchifier import OracleBatchifier
from src.guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from src.guesswhat.models.oracle.oracle_network import OracleNetwork
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



    parser.add_argument("-inference_mode", type=bool, default=False, help="inference mode True if you want to execute only test_dataset")


    args = parser.parse_args()
 
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    # logger.info("Save_path = ",save_path)
    # exit()


    logger = logging.getLogger()

    # Load config
    resnet_version = config['model']["image"].get('resnet_version', 50)
    finetune = config["model"]["image"].get('finetune', list())
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]
    use_glove = config["model"]["glove"]


    # Inference True if want to test the dataset_test of the pre-trained Weigth


    inference =  False       
    wait_inference = 1



    #############################
    #  LOAD DATA                
    #############################

    # Load image
    image_builder, crop_builder = None, None
    use_resnet = False
    logger.info("Loading ")
    t1 = time.time()

    if config['inputs'].get('image', False):
        logger.info('Loading images..')
        image_builder = get_img_builder(config['model']['image'], args.img_dir)
        use_resnet = image_builder.is_raw_image()

    if config['inputs'].get('crop', False):
        logger.info('Loading crops..')
        crop_builder = get_img_builder(config['model']['crop'], args.crop_dir, is_crop=True)
        use_resnet = crop_builder.is_raw_image()

    

    t2 = time.time()


    # Load data
    logger.info('Loading data..')

    all_img_bbox = {}
    all_img_describtion = []


    t1 = time.time()

    trainset =  OracleDataset.load(args.data_dir, "train",image_builder = image_builder, crop_builder = crop_builder,all_img_bbox  = all_img_bbox,all_img_describtion=all_img_describtion)
    validset =  OracleDataset.load(args.data_dir, "valid", image_builder= image_builder, crop_builder = crop_builder,all_img_bbox = all_img_bbox,all_img_describtion=all_img_describtion)
    testset  =  OracleDataset.load(args.data_dir, "test",image_builder= image_builder, crop_builder = crop_builder,all_img_bbox = all_img_bbox,all_img_describtion=all_img_describtion)

    
    t2 = time.time()


    logger.info("Time to load data = {}".format(t2-t1))

    # np.save("all_img_bbox.npy",all_img_bbox)
    # logger.info("Image_crop legnth= {}".format(len(all_img_describtion)))
    # logger.info("Image_crop = {}".format(all_img_describtion))
    # with open('all_img_bbox.json', 'a') as file:
    #         file.write(json.dumps(all_img_bbox,sort_keys=True, indent=4, separators=(',', ': ')))
    file_allquestion =  Path("all_question_game.txt")

    # verify if file exist

    if not file_allquestion.is_file():
        with open('all_question_game.txt', 'a') as file:
            for question in all_img_describtion:
                file.write(question+"\n")
    else:
        logger.info("all_question exist")                
    
    # Load dictionary
    logger.info('Loading dictionary Question..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file_question),dic_all_question="data/dict_word_indice.pickle")

    # Load dictionary
    tokenizer_description = None
    if config["inputs"]["description"]:
        logger.info('Loading dictionary Description......')
        tokenizer_description = GWTokenizer(os.path.join(args.data_dir,args.dict_file_description),question=False)

    # Build Network
    logger.info('Building network..')
    if tokenizer_description != None:
        network = OracleNetwork(config, num_words_question=tokenizer.no_words,num_words_description=tokenizer_description.no_words)
    else: 
        network = OracleNetwork(config, num_words_question=tokenizer.no_words,num_words_description=None)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config, finetune=finetune)
    best_param = network.get_predict()
    ##############################
    #  START  TRAINING           
    #############################
    logger.info("Start training .......")

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()
    resnet_saver = None

    logger.info("saver done !")

    cpu_pool = Pool(args.no_thread, maxtasksperchild=5000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    logger.info("gpu_options done !")


    # Retrieve only resnet variabes
    if use_resnet:
        resnet_saver = create_resnet_saver([network])
    
    # use_embedding = False
    # if config["embedding"] != "None":
    #      use_embedding = True

    logger.info("resnet_saver done !")


    glove = None
    if use_glove:
        logger.info('Loading glove..')
        glove = GloveEmbeddings(os.path.join(args.data_dir, config["glove_name"]),glove_dim=300,type_data="common_crow")


    logger.info("glove done !")   

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
        best_val_err = 0
        # best_train_err = None
        # # create training tools
        evaluator = Evaluator(sources, network.scope_name,network=network,tokenizer=tokenizer)

        # train_evaluator = MultiGPUEvaluator(sources, scope_names, networks=networks, tokenizer=tokenizer)
        # train_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)
        # eval_evaluator = Evaluator(sources, scope_names[0], network=networks[0], tokenizer=tokenizer)
        
        batchifier =  OracleBatchifier(tokenizer, sources, status=config['status'],glove=glove,tokenizer_description=tokenizer_description,args = args,config=config)

        stop_learning = False
        progress_compteur = 0
        t = 0


        if inference == False:

            while start_epoch < no_epoch and not stop_learning :

            # for t in range(start_epoch, no_epoch):
                logger.info('Epoch {}..'.format(t + 1))
                # logger.info('Epoch {}..'.format(t + 1))
                logger.info(" train_oracle | Iterator ...")
                
                t1 = time.time()
                train_iterator = Iterator(trainset,
                                        batch_size=batch_size, pool=cpu_pool,
                                        batchifier=batchifier,
                                        shuffle=True)

                t2 = time.time()

                logger.info(" train_oracle | Iterator...Total=".format(t2-t1))
                
                t1 = time.time()
                train_loss, train_accuracy = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer],out_net=best_param)
                t2 = time.time()

                logger.info(" train_oracle | evaluatorator...Total=".format(t2-t1))

                t1 = time.time()



                valid_iterator = Iterator(validset, pool=cpu_pool,
                                        batch_size=batch_size*2,
                                        batchifier=batchifier,
                                        shuffle=False)


                t2 = time.time()

                logger.info(" train_oracle | Iterator validset...Total=".format(t2-t1))


                t1 = time.time()
                # [network.get_emb_concat()]
                valid_loss, valid_accuracy = evaluator.process(sess, valid_iterator, outputs=outputs,type_data="Valid")
                t2 = time.time()

                logger.info(" train_oracle | evaluator ...Total=".format(t2-t1))

                logger.info("Training loss: {}".format(train_loss))
                logger.info("Training error: {}".format(1-train_accuracy))
                logger.info("Validation loss: {}".format(valid_loss))
                logger.info("Validation error: {}".format(1-valid_accuracy))
                
                t1 = time.time()

                if valid_accuracy > best_val_err:
                    best_train_err = train_accuracy
                    best_val_err = valid_accuracy
                    saver.save(sess, save_path.format('params.ckpt'))
                    progress_compteur = 0
                    logger.info("Oracle checkpoint saved...")
                    pickle_dump({'epoch': t}, save_path.format('status.pkl'))


                elif valid_accuracy < best_val_err:
                    progress_compteur += 1 
                

                if int(progress_compteur) == int(wait_inference):
                    stop_learning = True

                t2 = time.time()
                logger.info(" train_oracle | Condition ...Total=".format(t2-t1))

                t += 1
                start_epoch += 1


        # Load early stopping

        t1 = time.time()
        if inference:
            # save_path = "out/oracle/46499510c2ab980278d91eeff89aa06f/{}" # 
            # save_path = "out/oracle/9efb52e0bd872e1f4e64f66b35a2f092/{}" # question
            # save_path = "out/oracle/a9cc5b30b2024399c79b6997086c5265/{}" # question,category,spatial
            # save_path = "out/oracle/89570bad275ddde7b69a5c37659bd40e/{}" # question,category,spaticial,crop
            # save_path = "out/oracle/b158b76a46173ff33e4aec021e267e5a/{}" # question,category,spaticial,history
            # save_path = "out/oracle/30ef7335e38c93632b58e91fa732cf2d/{}" # question,category,spaticial,history,Images
            # save_path = "out/oracle/d9f1951536bbd147a3ea605bb3cbdde7/{}" # question,category,spaticial,history,Crop                                                             # question,category,spaticial,history,Crop 
            # save_path = "out/oracle/4a9f62698e3304c4c2d733bff0b24ee2/{}"
            save_path =   "out/oracle/a630385c990e5cc470c2488a244f18dc/{}"

            # out/oracle/ce02141129f6d87172cafc817c6d0b59/params.ckpt
            # save_path = save_path.format('params.ckpt')

            logger.info("***** save_path = ".format(save_path))
            




        save_path = save_path.format('params.ckpt')
        saver.restore(sess, save_path)


        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=True)


        logger.info("Output = {}".format(outputs[1]))
        logger.info("Best_param = {}".format(best_param))


        test_loss, test_accuracy = evaluator.process(sess, test_iterator,  outputs=outputs ,out_net=best_param,inference=inference,type_data="Test")
       
        t2 = time.time()
        
        logger.info(" train_oracle | Iterator testset  ...Total=".format(t2-t1))
       
        try:
            logger.info("Testing loss: {}".format(test_loss))
        except Exception:
            logger.info("Erreur loss")

        try:
            logger.info("Testing error: {}".format(1-test_accuracy))
        except Exception:
            logger.info("Erreur accuracy")
