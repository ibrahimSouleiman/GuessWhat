#!/usr/bin/env python
import numpy
import os
import tensorflow as tf
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import h5py

from generic.data_provider.nlp_utils import DummyTokenizer
from generic.data_provider.iterator import Iterator

def extract_features(
        img_input,
        ft_output,
        network_ckpt, 
        dataset_cstor,
        dataset_args,
        batchifier_cstor,
        out_dir,
        set_type,
        batch_size,
        no_threads,
        gpu_ratio):

    # CPU/GPU option
    cpu_pool = Pool(no_threads, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)

    # gpu_options.allow_growth = True


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:

        saver = tf.train.Saver()
        saver.restore(sess, network_ckpt)

        # tf.initialize_all_variables().run()
        for one_set in set_type:
    
            print("Load dataset -> set: {}".format(one_set))
            dataset_args["which_set"] = one_set
            dataset = dataset_cstor(**dataset_args)
    
            # hack dataset to only keep one game by image
            image_id_set = {}
            games = []

            nb_total_trouve = 0
            nb_nonTrouve = 0

            print("[Data Length ] = {}".format(len(dataset.games)))
            for game in dataset.games:
                if game.image.id not in image_id_set:
                    games.append(game)
                    image_id_set[game.image.id] = 1

            #
            #         img = game.image.url.split("/")[-1]
            #         img = img + ".jpg"
            #
            #         print("[Img]= {}".format(img))
            #
            #         s = open(os.path.join("data/img/raw/", "42.jpg"))
            #
            #         try:
            #             print(str(img++".jpg"))
            #             f = open(os.path.join("data/img/raw/", img))
            #
            #             nb_total_trouve += 1
            #         except:
            #             # print("Not Found !!")
            #             nb_nonTrouve += 1
            # print("[Trouve = {} et Non_Trouve = {} ]".format(nb_total_trouve,nb_nonTrouve))
            # # print(img ++".jpg")









            dataset.games = games
            no_images = len(games)
    
            source_name = os.path.basename(img_input.name[:-2])
            dummy_tokenizer = DummyTokenizer()
            batchifier = batchifier_cstor(tokenizer_question=dummy_tokenizer, sources=[source_name])
            iterator = Iterator(dataset,
                                batch_size=batch_size,
                                pool=cpu_pool,
                                batchifier=batchifier)


            ############################
            #  CREATE FEATURES
            ############################

            print("Start computing image features...")
            filepath = os.path.join(out_dir, "{}_features.h5".format(one_set))

            with h5py.File(filepath, 'w') as f:
                print("--- 1")
                ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]
                print("--- 2")
                ft_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
                print("--- 3")
                idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
                print("--- 4")
                pt_hd5 = 0

                print("--- 5")

                for batch in tqdm(iterator):
                    print("--- 6")
                    print(" ... ",numpy.array(batch[source_name]),numpy.array(batch[source_name]).shape)
                    feat = sess.run(ft_output, feed_dict={img_input: numpy.array(batch[source_name])})
    
                    # Store dataset
                    print("--- 7")
                    batch_size = len(batch["raw"])
                    ft_dataset[pt_hd5: pt_hd5 + batch_size] = feat
    
                    print("--- 8")

                    # Store idx to image.id
                    for i, game in enumerate(batch["raw"]):
                        idx2img[pt_hd5 + i] = game.image.id
    
                    print("--- 9")
                    # update hd5 pointer
                    pt_hd5 += batch_size
                print("Start dumping file: {}".format(filepath))
            print("Finished dumping file: {}".format(filepath))
    
    
    print("Done!")
