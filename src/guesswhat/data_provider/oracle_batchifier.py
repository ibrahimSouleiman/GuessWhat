import numpy as np
import collections
from PIL import Image

from generic.data_provider.batchifier import AbstractBatchifier

from generic.data_provider.image_preprocessors import get_spatial_feat, resize_image
from generic.data_provider.nlp_utils import padder,padder_3d


import time

answer_dict = \
    {  'Yes': np.array([1, 0, 0], dtype=np.int32),
       'No': np.array([0, 1, 0], dtype=np.int32),
       'N/A': np.array([0, 0, 1], dtype=np.int32)
    }

class OracleBatchifier(AbstractBatchifier):

    def __init__(self, tokenizer_question, sources,tokenizer_description = None ,embedding=None, status=list()):
        self.tokenizer_question = tokenizer_question
        self.tokenizer_description = tokenizer_description
        self.sources = sources
        self.status = status
        self.embedding=embedding

    def filter(self, games):
        if len(self.status) > 0:
            return [g for g in games if g.status in self.status]
        else:
            return games


    def apply(self, games):
        sources = self.sources
        

        t1 = time.time()
        batch = collections.defaultdict(list)

        for i, game in enumerate(games):
            batch['raw'].append(game)

            image = game.image
            question = self.tokenizer_question.apply(game.questions[0])
            
            batch['question'].append(question)
            # 

            if 'embedding_vector' in sources:
                
                assert  len(game.questions) == 1
                # Add glove vectors (NB even <unk> may have a specific glove)
                # print("oracle_batchifier | question = {}".format(game.questions))
                words = self.tokenizer_question.apply(game.questions[0],tokent_int=False)
                # print(" End ................... ,",words)
                # print(len(words[0]))
                # print(words)
                # print("/////////// embedding_vector=")
                if type(words) == int:
                    exit()
                
                if "question_pos" in sources:
                    # print("/////////// question_pos")
                    embedding_vectors,embedding_pos = self.embedding.get_embeddings(words) # slow (copy gloves in process)
                    # print("..... question_pos............. embedding_vectors",len(embedding_vectors[0]))

                    batch['embedding_vector'].append(embedding_vectors)
                    batch['embedding_vector_pos'].append(embedding_pos)
                    batch['question_pos'].append(question)

                    # print(batch['embedding_vector_pos'])
                else:
                    # print("/////////// question_pos NOT EXIST")

                    embedding_vectors,_ = self.embedding.get_embeddings(words) # slow (copy gloves in process)

                    # print("////////// embedding_vectors=",len(embedding_vectors[0]))
                    batch['embedding_vector'].append(embedding_vectors)

                # print(" Oracle_batchifier | embedding_vector= {}".format(embedding_vectors))

                # print("---- Embedding = ",len(embedding_vectors))
                # print("----  =",len(embedding_vectors[0]))

                




                # games.question = ['am I a person?'],            
                # batch['question'].append(self.tokenizer_question.apply(game.questions[0]))

            if 'description' in sources:
                

                description = self.tokenizer_question.apply(game.image.description)
       
                batch['description'].append(description)

                # print("/////////// question_pos")
                embedding_vectors,_ = self.embedding.get_embeddings(words,description=True) # slow (copy gloves in process)

                # print("...........description....... embedding_vectors",len(embedding_vectors[0]))
                batch['embedding_vector_description'].append(embedding_vectors)
               


            if 'answer' in sources:
                assert len(game.answers) == 1
                batch['answer'].append(answer_dict[game.answers[0]])

            if 'category' in sources:
                batch['category'].append(game.object.category_id)

            if 'allcategory' in sources:
                allcategory = []
                allcategory_hot = np.zeros(shape=(90),dtype=int)
                # print("Oracle_batchifier |  Allcategory -------------------------------")
                for obj in game.objects:
                    allcategory.append(obj.category_id - 1)



                allcategory_hot[allcategory] = 1
                # print("...   ",allcategory,allcategory_hot)

                batch['allcategory'].append(allcategory_hot)

            if 'spatial' in sources:
                spat_feat = get_spatial_feat(game.object.bbox, image.width, image.height)
                batch['spatial'].append(spat_feat)

            if 'crop' in sources:
                batch['crop'].append(game.object.get_crop())

            if 'image' in sources:
                batch['image'].append(image.get_image())

            if 'mask' in sources:
                assert "image" in batch['image'], "mask input require the image source"
                mask = game.object.get_mask()
                ft_width, ft_height = batch['image'][-1].shape[1],\
                                     batch['image'][-1].shape[2] # Use the image feature size (not the original img size)
                mask = resize_image(Image.fromarray(mask), height=ft_height, width=ft_width)
                batch['mask'].append(mask)

        # padding = self.embedding.get_embeddings(["<padding>"])[0]
        # print("padding | = {}".format(padding))

        # pad the questions
        
    
        

        batch['question'], batch['seq_length_question'] = padder(batch['question'],
                                                        padding_symbol=self.tokenizer_question.padding_token)

        if "question_pos" in sources:
            batch['question_pos'], batch['seq_length_pos'] = padder(batch['question_pos'],
                                                            padding_symbol=self.tokenizer_question.padding_token)

        if "description" in sources:
            batch['description'], batch['seq_length_description'] = padder(batch['description'],
                                                            padding_symbol=self.tokenizer_description.padding_token)

        # batch['embedding_vector_pos'], _ = padder_3d(batch['embedding_vector_pos'])
        if 'embedding_vector' in sources:
                        batch['embedding_vector'], _ = padder_3d(batch['embedding_vector'])

        if 'embedding_vector_pos' in sources:
                        batch['embedding_vector_pos'], _ = padder_3d(batch['embedding_vector_pos'])

        if 'embedding_vector_description' in sources:
                        batch['embedding_vector_description'],_ = padder_3d(batch['embedding_vector_description'])


        



        # if 'description' in sources:
        #     # complete par padding en prenons la taille maximal
        # batch['description'], batch['seq_length_description'] = padder_3d(batch['description'])


        t2 = time.time()

        total = t2-t1


        # print("finish oracle_bachifier .... time=",total)
        return batch



