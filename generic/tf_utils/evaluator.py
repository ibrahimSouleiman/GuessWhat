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

    def process(self, sess, iterator, outputs, out_net=None,inference=False, listener=None,type_data="train"):

        assert isinstance(outputs, list), "outputs must be a list"


        original_outputs = list(outputs)
        

        is_training = any([is_optimizer(x) for x in outputs])

        # if listener is not None:
        #     outputs += [listener.require()]  # add require outputs
        #     # outputs = flatten(outputs)     # flatten list (when multiple requirement)
        #     outputs = list(OrderedDict.fromkeys(outputs))  # remove duplicate while preserving ordering
        # listener.before_epoch(is_training)

        n_iter = 1.
        aggregated_outputs = [0.0 for v in outputs if is_scalar(v) and v in original_outputs]

        good_predict = {}
        bad_predict = {}
        image_id = None

        self.good_predict = 0
        self.bad_predict = 0

        for batch in tqdm(iterator):
            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training

            if inference:
                
                image_id = batch["image_id"]
              
                batch_1 = {}
                batch_2 = {}

                question= batch["question_word"]
                images_id = batch["image_id"]
                # crops_id = batch["crop_id"]

                for key,values in batch.items():
                    if  key!="question_word" and key!= "image_id":
                        pass
                    elif type(values) != bool :
                        batch_1[key] = [values[0]]
                        batch_2[key] = [values[1]]
                    else:
                        batch_1[key] = values
                        batch_2[key] = values

            
            if inference == False:

                batch = {key:value for key,value in batch.items() if  key!="question_word" and key!= "image_id"}
                results = self.execute(sess, outputs,batch,type_data )


            if inference:

                

                old_batch = batch
                batch = {key:value for key,value in batch.items() if  key!="question_word" and key!= "image_id"}
                prediction = self.execute(sess,out_net,batch, type_data)
                results = self.execute(sess, outputs,batch, type_data )

            
                # for question,gold,prediction in zip(old_batch["question"],old_batch["answer"],prediction):
                #     gold_argmax = np.argmax(gold)
                #     predict_argmax = np.argmax(prediction)
                #     self.good_predict += 1

                #     if gold_argmax == predict_argmax:
                #         self.good_predict += 1
                #         print("GOOD | Image_id ={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict = {}".format(image_id,question,categories_object,gold_argmax,predict_argmax,prediction) )
                #         # print("GOOD |  Question= {},  gold={}, prediction={}, proba_predict = {}".format(question,gold_argmax,predict_argmax,prediction) )

                #     else:
                #         self.bad_predict += 1
                #         print("BAD | Image_id ={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict = {}".format(image_id,question,categories_object,gold_argmax,predict_argmax,prediction) )
                #         # print("BAD |  Question= {},  gold={}, prediction={}, proba_predict = {}".format(question,gold_argmax,predict_argmax,prediction) )

                # exit()
                print(".... old__batch={} ".format(old_batch))

                for image_id,question,categories_object,gold,prediction in zip(old_batch["image_id"],old_batch["question_word"],old_batch["category"],old_batch["answer"],prediction):
                    print("....... image ..............")

                    gold_argmax = np.argmax(gold)
                    predict_argmax = np.argmax(prediction)
                    self.good_predict += 1
                    if gold_argmax == predict_argmax:
                        self.good_predict += 1
                        print("GOOD | Image_id ={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict = {}".format(image_id,question,categories_object,gold_argmax,predict_argmax,prediction) )
                    else:
                        self.bad_predict += 1
                        print("BAD | Image_id ={}, Question= {}, Categorie_object={}, gold={}, prediction={}, proba_predict = {}".format(image_id,question,categories_object,gold_argmax,predict_argmax,prediction) )



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
        

        print(" Result good_predict = {} , bad_predict={}".format(self.good_predict,self.bad_predict))

        return aggregated_outputs
            


    def execute(self, sess, output, batch,type_data = "Train"):
        #print("+++++++++++++++++++++",batch.items())
        feed_dict = {self.scope +key + ":0": value for key, value in batch.items() if key in self.provided_sources}
        # print("-- [{}]  question = {}--".format(type_data, feed_dict["oracle/question:0"]))
        # print("shape = {} ".format(np.asarray(feed_dict["oracle/question:0"]).shape ))

        # for key,value in feed_dict.items():
        #     print("-- [{}]  Feed_Dict = {} : {}-- ".format(type_data,key,np.asarray(value).shape )) 
        
        # exit()
        # print("answer = {}".format(feed_dict["oracle/answer:0"]))
        
        # print("-- [{}]  image_values = {}--".format(type_data, feed_dict["oracle/image:0"]))
        # print("========================================================================")
        # print("-- [{}] Output = {} ".format(type_data, output))
        # print("-- [{}] result = {} ".format(type_data,sess.run(output, feed_dict=feed_dict)))
        # print("")
        

        if type_data == "Valid":
            # out = output[0].eval(feed_dict = feed_dict)
            # print("Sess_emb = {}".format(out))
            # print("Shape = {}".format(out.shape))
            # print("Other = {}".format(tf.get_variable("oracle/l2_normalize:0").eval(feed_dict = feed_dict)))
            # print("ouput = {}".format(sess.run(output[1], feed_dict=feed_dict)))
            # exit()
            pass
        # print("resultat_softmax = {}".format(sess.run(tf.get_variable("oracle/mlp/Softmax_1:0"), feed_dict=feed_dict)))
        
        
        # exit()

        
        return sess.run(output, feed_dict=feed_dict)



class MultiGPUEvaluator(object):
    """
    Wrapper for evaluating on multiple GPUOptions

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