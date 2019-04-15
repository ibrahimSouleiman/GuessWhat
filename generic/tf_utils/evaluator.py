from tqdm import tqdm
from numpy import float32
import copy
import os
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np

# TODO check if optimizers are always ops? Maybe there is a better check
def is_optimizer(x):
    return hasattr(x, 'op_def')

def is_summary(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.string


def is_float(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32


def is_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


class Evaluator(object):
    def __init__(self, provided_sources, scope="", writer=None,
                 network=None, tokenizer=None): # debug purpose only, do not use in the code

        self.provided_sources = provided_sources
        self.scope = scope
        self.writer = writer
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # Debug tools (should be removed on the long run)
        self.network=network
        self.tokenizer = tokenizer

    def process(self, sess, iterator, outputs, out_net=None,inference=False, listener=None):

        assert isinstance(outputs, list), "outputs must be a list"

        original_outputs = list(outputs)
        

        is_training = any([is_optimizer(x) for x in outputs])

        # if listener is not None:
        #     outputs += [listener.require()]  # add require outputs
        #     # outputs = flatten(outputs) # flatten list (when multiple requirement)
        #     outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering


        # listener.before_epoch(is_training)

        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v) and v in original_outputs]

        good_predict = {}
        bad_predict = {}

        for batch in tqdm(iterator):
            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training

            if inference:
                # print(batch,".............. batch")
                # evaluate the network on the batch
                # print("Batch_Length=",len(batch["embedding_vector_ques"]))
                print("Dict_keys=",batch.keys())
                id_images = batch["image_id"]
                # print(id_images)
                # exit()
                batch_1 = {}
                batch_2 = {}

                question= batch["question"]
                images_id = batch["image_id"]
                crops_id = batch["crop_id"]



                for key,values in batch.items():
                    if key=="question" or key=="image_id" or key=="crop_id" :
                        pass
                    elif type(values) != bool :
                        batch_1[key] = [values[0]]
                        batch_2[key] = [values[1]]
                    else:
                        batch_1[key] = values
                        batch_2[key] = values

        


                # print("true_1=",batch_1["answer"])
                # print("true_2=",batch_2["answer"])

                # print("Categorie_1 =",batch_1["category"])
                # print("Categorie_1 =",batch_1["category"])
                # print("Batch=",batch["embedding_vector_ques"])


                

                # resul_net_1 = self.execute(sess, out_net, batch_1)
                # resul_net_2 = self.execute(sess, out_net, batch_2)
                # print(np.array_equal(np.asarray(resul_net_1),np.asarray(resul_net_2)))

                # if resul_net_1!= [0] or resul_net_2 != [0]:
                #     print(resul_net_1)
                #     print(resul_net_2)
                #     exit()

                # elif np.array_equal(np.asarray(resul_net_1),np.asarray(resul_net_2)) == False:
                #     print("Resultat 1= ",resul_net_1)
                #     # resul_net = self.execute(sess, out_net, batch)
                #     print("Resultat 2= ",resul_net_2)
                # else:
                #     print("Resultat = ",resul_net_1,resul_net_2)

                    # exit()
                # process the results
                # out = tf.get_variable("oracle/mlp/Softmax_1:0")
                # r = self.execute(sess, out, batch)
                # batch = {key:value for key,value in batch.items() if key!="question"}

                
            batch = {key:value for key,value in batch.items() if key!="question" and key!="image_id" and key!="crop_id"}
            
            results = self.execute(sess, outputs,batch )
            output_1 = self.execute(sess,out_net, batch_1)
            output_2 = self.execute(sess,out_net, batch_2)

            out_1 = np.argmax(output_1[0])
            out_2 = np.argmax(output_2[0])
            print(batch_1["answer"])
            gold_1 = np.argmax(batch_1["answer"][0])
            gold_2 = np.argmax(batch_1["answer"][0])

            if inference:
                if gold_1 == out_1:
                    print("GOOD | Image_id ={}, Crop_id={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict ={}".format(images_id[0],crops_id[0],question[0],batch_1["category"][0],gold_1,out_1,output_1[0]) )
                else:
                    print("BAD | Image_id ={}, Crop_id={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict ={}".format(images_id[0],crops_id[0],question[0],batch_1["category"][0],gold_1,out_1,output_1[0]) )
                if gold_2 == out_2:
                    print("GOOD | Image_id ={}, Crop_id={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict ={}".format(images_id[1],crops_id[1],question[1],batch_2["category"][0],gold_2,out_2,output_2[0]) )
                else:
                    print("BAD |Image_id ={}, Crop_id={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict ={}".format(images_id[1],crops_id[1],question[1],batch_2["category"][0],gold_2,out_2,output_2[0]) )

                # print(results)

            # result = self.execute(sess, out_net,batch )
            # result = sess.run(out_net,batch)


            i = 0
            for var, result in zip(outputs, results):
                if is_scalar(var) and var in original_outputs:
                    # moving average
                    aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                    i += 1
                elif is_summary(var):  # move into listener?
                    self.writer.add_summary(result)

                if listener is not None and listener.valid(var):
                    listener.after_batch(result, batch, is_training)

            n_iter += 1
            
            

        if listener is not None:
            listener.after_epoch(is_training)
        
        aggregated_outputs=[None,None]

        return aggregated_outputs
            
    # def process(self, sess, iterator, outputs, listener=None):

    #     assert isinstance(outputs, list), "outputs must be a list"

    #     original_outputs = list(outputs)

    #     is_training = any([is_optimizer(x) for x in outputs])

    #     if listener is not None:
    #         outputs += [listener.require()]  # add require outputs
    #         # outputs = flatten(outputs) # flatten list (when multiple requirement)
    #         outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
    #         listener.before_epoch(is_training)

    #     n_iter = 1.
    #     aggregated_outputs = [0.0 for v in outputs if is_scalar(v) and v in original_outputs]

    #     #print(" Evaluator | iterator =",iterator)

    #     for batch in tqdm(iterator):

            
    #         # Appending is_training flag to the feed_dict
    #         batch["is_training"] = is_training

    #         # evaluate the network on the batch
    #         results = self.execute(sess, outputs, batch)
    #         # process the results
    #         i = 0
    #         for var, result in zip(outputs, results):
    #             if is_scalar(var) and var in original_outputs:
    #                 # moving average
    #                 aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
    #                 i += 1
    #             elif is_summary(var):  # move into listener?
    #                 self.writer.add_summary(result)

    #             if listener is not None and listener.valid(var):
    #                 listener.after_batch(result, batch, is_training)

    #         n_iter += 1

    #     if listener is not None:
    #         listener.after_epoch(is_training)

    #     return aggregated_outputs

    def execute(self, sess, output, batch):
        #print("+++++++++++++++++++++",batch.items())
        feed_dict = {self.scope +key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        #print("-- Feed_Dict = {}--",feed_dict.keys())
        #print("------Output----- ===",output)
        # exit()

        
        return sess.run(output, feed_dict=feed_dict)



class MultiGPUEvaluator(object):
    """Wrapper for evaluating on multiple GPUOptions

    parameters
    ----------
        provided_sources: list of sources
            Each source has num_gpus placeholders with name:
            name_scope[gpu_index]/network_scope/source
        network_scope: str
            Variable scope of the model
        name_scopes: list of str
            List that defines name_scope for each GPU
    """

    def __init__(self, provided_sources, name_scopes, writer=None,
                 networks=None, tokenizer=None): #Debug purpose only, do not use here

        # Dispatch sources
        self.provided_sources = provided_sources
        self.name_scopes = name_scopes
        self.writer = writer

        self.multi_gpu_sources = []
        for source in self.provided_sources:
            for name_scope in name_scopes:
                self.multi_gpu_sources.append(os.path.join(name_scope, source))


        # Debug tools, do not use in the code!
        self.networks = networks
        self.tokenizer = tokenizer



    def process(self, sess, iterator, outputs, listener=None):

        assert listener is None, "Listener are not yet supported with multi-gpu evaluator"
        assert isinstance(outputs, list), "outputs must be a list"

        # check for optimizer to define training/eval mode
        is_training = any([is_optimizer(x) for x in outputs])

        # Prepare epoch
        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v)]


        scope_to_do = list(self.name_scopes)
        multi_gpu_batch = dict()

        for batch in tqdm(iterator):

            assert len(scope_to_do) > 0

            # apply training mode
            batch['is_training'] = is_training

            # update multi-gpu batch
            name_scope = scope_to_do.pop()
            for source, v in batch.items():
                multi_gpu_batch[os.path.join(name_scope, source)] = v



            if not scope_to_do: # empty list -> multi_gpu_batch is ready!
                n_iter += 1
                # Execute the batch
                results = self.execute(sess, outputs, multi_gpu_batch)

                # reset mini-batch
                scope_to_do = list(self.name_scopes)
                multi_gpu_batch = dict()

                # process the results
                i = 0
                for var, result in zip(outputs, results):
                    if is_scalar(var) and var in outputs:
                        # moving average
                        aggregated_outputs[i] = ((n_iter - 1.) / n_iter) * aggregated_outputs[i] + result / n_iter
                        i += 1

                    elif is_summary(var):  # move into listener?
                        self.writer.add_summary(result)

                    # No listener as "results" may arrive in different orders... need to find a way to unshuffle them

        return aggregated_outputs


    def execute(self, sess, output, batch):
        feed_dict = {key + ":0": value for key, value in batch.items() if key in self.multi_gpu_sources}
        return sess.run(output, feed_dict=feed_dict)