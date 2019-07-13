import gzip
import json
import copy
import os
import nltk
import time
import numpy as np
import logging


from collections import Counter
from matplotlib import pyplot as plt

from PIL import ImageFont, ImageDraw
from PIL import Image as PImage
from PIL import Image as PIL_Image

from nltk.tokenize import TweetTokenizer
from nltk import WordNetLemmatizer

from generic.data_provider.dataset import AbstractDataset
from draw.Generate_Category import get_category
from generic.data_provider.nlp_utils import Embeddings,get_embeddings

# TODO find a cleaner way!

# from cocoapi.PythonAPI.pycocotools import mask as cocoapi
# from cocoapi.PythonAPI.pycocotools import mask as cocoapi
logger = logging.getLogger()

try:
    import cocoapi.PythonAPI.pycocotools.mask as cocoapi
    use_coco = True
except ImportError:
    logger.info("Coco API was not detected - advanced segmentation features cannot be used")
    use_coco = False
    pass


all_word = []


class Game:

    def __init__(self, id, object_id, image, objects, qas , status, which_set, image_builder, crop_builder,lemmatizer,all_img_bbox,all_img_describtion,embedding=None):
        self.dialogue_id = id
        self.object_id = object_id
        
        compteur = 0

        # logger.info("image_width = {},image_height = {}".format(image["width"],image["height"]))

        self.image = Image(id=image["id"],
                           width=image["width"],
                           height=image["height"],
                           url=image["coco_url"],
                           description=image["description"],
                           which_set=which_set,
                           image_builder=image_builder)
        

        self.objects = []
        # self.all_category = []
     
        # logger.info("Object in Image = {} ".format(len(objects)))

        for o in objects:
            # self.all_category.append(o['category'])
            new_obj = Object(id=o['id'],
                             category=o['category'],
                             category_id=o['category_id'],
                             bbox=Bbox(o['bbox'], image["width"], image["height"]),
                             area=o['area'],
                             segment=o['segment'],
                             crop_builder=crop_builder,
                             which_set=which_set,
                             image=self.image)

         
        
            self.objects.append(new_obj)


            # img = new_obj.get_crop()
            # img = [x / 255.0 for x in img]
            
            # img = np.asarray(img)
            # logger.info("Img A= {}".format(img.shape))
            # exit() 
            # # logger.info("Type = {} =".format(type(img)))
            # # logger.info("Type = {} =".format(img.shape))
            # img = np.reshape(img,224*224*3)
            # logger.info("Img = {}".format(img.shape))
            # img = np.reshape(img,(224,224,3))
            # plt.imshow(img,cmap='gray')
            # plt.title("Other")
            # plt.show()


            if o['id'] == object_id:
                self.object = new_obj  # Keep ref on the object to find
                # logger.info("image_id = {},{}".format(image["id"],type(image["id"])))

                # if image["id"] == 632:
                #     img = new_obj.get_crop()
                #     img = [x / 255.0 for x in img] 
                #     img = np.asarray(img)
                #     # # logger.info("Type = {} =".format(type(img)))
                #     # # logger.info("Type = {} =".format(img.shape))
                #     # img = np.reshape(img,224*224*3)
                #     # logger.info("Img = {}".format(img.shape))
                #     # img = np.reshape(img,(224,224,3))
                #     logger.info("Img = {} ".format(img.shape))
                #     exit()
                #     plt.imshow(img,cmap='gray')
                #     plt.title("Crop")
                #     plt.show()
       




                # all_img_describtion [image["id"]]  = img   

                # logger.info("object = {} ".format(self.object))


                # all_img_describtion[image["id"]] = [new_obj]

                # logger.info("Img = {}".format(img))

                # plt.imshow(img,cmap='gray')
                # plt.show()

                # logger.info("shape = {}".format(img.shape))
                # exit()
                # all_img_bbox[image["id"] ]= o['bbox']    
                # logger.info("Select=",o['category'])
            # else:
            #     pass


                # logger.info(o['category']) 
        # exit()    

        # logger.info("-----------------------")
        # compteur += 1
        # if compteur == 15:
        #     exit()
        self.question_ids = [qa['id'] for qa in qas]
        self.questions = [qa['question'] for qa in qas]
        _ = [all_img_describtion.append(qa['question']) for qa in qas]
        self.answers = [qa['answer'] for qa in qas]
        self.status = status
        self.last_question = ["unk" for i in range(1)]
        self.all_last_question = [self.last_question for i in range(6)]


    def show(self, img_raw_dir, display_index=False, display_mask=False):
            image_path = os.path.join(img_raw_dir, self.image.filename)
            img = PImage.open(image_path)
            draw = ImageDraw.Draw(img)

            for i, obj in enumerate(self.objects):
                if display_index:
                    draw.text((obj.bbox.x_center, self.image.height-obj.bbox.y_center), str(i))
                if display_mask:
                    logger.info("Show mask: Not yet implemented... sry")

            img.show()



class Word:
    """
    classe qui contient pour chaque mot , son lemme sa parti de discourt
    """
    def __init__(self,word,lemme=False,pos=False,lemmatizer=None):

        self.word = word
        self.lemme = ""
        self.pos = ""
        self.lemmetizer = lemmatizer
        if self.lemmetizer != None:
            if lemme :
                self.lemme = self.lemmetizer.lemmatize(word)
            
        if pos:
            self.pos = nltk.pos_tag([word])

    def get_word(self):
        return self.word

    def get_lemme(self):
        return self.lemme
    
    def get_pos(self):
        return self.pos

    def get_all(self):
        return self.word,self.lemme,self.pos
    

class Image:
    def __init__(self, id, width, height, url,description, which_set, image_builder=None):
        self.id = id
        self.width = width
        self.height = height
        self.url = url
        self.description = description
        self.image_loader = None
        if image_builder is not None:
            self.filename = "{}.jpg".format(id)
            self.image_loader = image_builder.build(id, which_set=which_set, filename=self.filename, optional=False)



    def get_idimage(self):
        return self.id


    def get_image(self, **kwargs):
        if self.image_loader is not None:
            return self.image_loader.get_image(**kwargs)
        else:
            return None




class Bbox:
    def __init__(self, bbox, im_width, im_height):
        # Retrieve features (COCO format)
        # logger.info("bbox = {} ,im_width = {} ,im_height = {}".format(bbox,im_width,im_height))
        # exit()


        self.x_width = bbox[2]
        self.y_height = bbox[3]

        self.x_left = bbox[0]
        self.x_right = self.x_left + self.x_width

        self.y_upper = im_height - bbox[1]
        self.y_lower = self.y_upper - self.y_height

        self.x_center = self.x_left + 0.5*self.x_width
        self.y_center = self.y_lower + 0.5*self.y_height

        self.coco_bbox = bbox

    def __str__(self):
        return "center : {0:5.2f}/{1:5.2f} - size: {2:5.2f}/{3:5.2f}"\
            .format(self.x_center, self.y_center, self.x_width, self.y_height)


class Object:

    def __init__(self, id, category, category_id, bbox, area, segment, crop_builder, image, which_set):
        self.id = id
        self.category = category
        self.category_id = category_id
        self.bbox = bbox
        self.area = area
        self.segment = segment
 
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        self.rle_mask = None
        if use_coco:
            self.rle_mask = cocoapi.frPyObjects(self.segment,
                                                h=image.height,
                                                w=image.width)


        # logger.info("Guess_What = {} ".format(self.rle_mask))
        # logger.info("get_mask_shape A= {} ".format(self.get_mask().shape ))
        # get_mask_or = self.get_mask()
        # get_mask = np.reshape(get_mask_or,(360,640))
        # logger.info("get_mask_shape B= {} ".format(get_mask.shape))

        # img_path = os.path.join("data/img/raw", "{}.jpg".format(image.id))
        # img = PIL_Image.open(img_path).convert('RGB')
        # # img = resize_image(img, self.width , self.height)
        
        # # logger.info("/*/*/*/*/******* Image ********/*/*/*/*/*/*/*/ {}".format(self.img_path))
        
        
        # # logger.info("img_shape = {} {} ".format(self.width,self.height))
        # img_segment = np.multiply(img,get_mask_or)


        # plt.imshow(img_segment)
        # # plt.imshow(get_mask)
        # plt.show()
    

        if crop_builder is not None:
            filename = "{}.jpg".format(image.id)
            # logger.info("id = {} ,filemane = {} ,which_set = {},bbox = {}".format(id,filename,which_set,bbox))
            


            self.crop_loader = crop_builder.build(id, filename=filename, which_set=which_set, bbox=bbox)            # logger.info("Image_id=",image.id)
            # logger.info("crop_loader = {} ".format(self.crop_loader))
            # exit()


    def get_mask(self):
        assert self.rle_mask is not None, "Mask option are not available, please compile and link cocoapi (cf. cocoapi/PythonAPI/setup.py)"
        return cocoapi.decode(self.rle_mask)

    def get_idObject(self):
        return self.id

    def get_crop(self, **kwargs):
        assert self.crop_loader is not None, "Invalid crop loader"
        return self.crop_loader.get_image(**kwargs)




class Dataset(AbstractDataset):

    """Loads the dataset."""
    
    def __init__(self, folder, which_set, image_builder=None, crop_builder=None,all_img_bbox=None,all_img_describtion=None):
        self.set = which_set
        self.lemmas = WordNetLemmatizer()
        self.maxlength_question = 0
        self.count_questions = {}
        self.indice = 0
        self.total = 0
        
        filegues = '{}/guesswhat_{}2014.jsonl.gz'.format(folder,which_set)
        file_description = '{}/captions_{}2014.json'.format(folder, which_set)
        games = []
        nb_pass = 0
        nb_erreur = 0

        t1 = time.time()
        all_size = []

        with gzip.open(filegues) as f:
            for i,line in enumerate(f):
                line = line.decode("utf-8")
                game = json.loads(line.strip('\n'))
                # try : 
                    # logger.info(game["image"])
                    # logger.info("Dans try")
                    # logger.info(game["id"],type(game["id"]))

                nb_pass += 1

                g = Game(id=game['id'],
                        object_id=game['object_id'],
                        objects=game['objects'],
                        qas=game['qas'],
                        image=game['image'],
                        status=game['status'],
                        which_set=which_set,
                        image_builder=image_builder,
                        crop_builder=crop_builder,
                        lemmatizer = self.lemmas,
                        all_img_bbox = all_img_bbox,
                        all_img_describtion = all_img_describtion,
                        )

                question_length = len(g.questions)
                

                # self.count_questions[len(self.count_questions)] = question_length

                self.total += question_length 
                
                for question in g.questions:
                    words = question.split()
                    all_size.append(len(words))

        
                if self.maxlength_question < question_length: self.maxlength_question = question_length
                games.append(g)

                if len(games) > 50: break

        logger.info(" Max Length of Question = {} , total_question = {}, nb_parties = {} | {}".format(self.maxlength_question,self.total,len(games),which_set))


       

        print("Dataset ...")
        super(Dataset, self).__init__(games)


class OracleDataset(AbstractDataset):
    """
    Each game contains a single question answer pair
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        self.compteur = 0
        self.all_category = []
        self.categories_questions = {0:0,1:0,2:0,3:0}
        self.all_question = []
        # self.embedding = Embeddings(file_name=["ted_en-20160408.zip" ],embedding_name="fasttext",emb_dim=100)
        # self.words = []
        self.length_question = {}
        self.tokenizer = TweetTokenizer(preserve_case=True)


        for i,g in enumerate(old_games):
            game,words = self.split(g)
            # self.words += words 
            new_games += game


        # self.words = list(set(self.words))

      



        # logger.info(" Nb question = ",format(self.compteur))
        # exit()
        # logger.info(" Guess_dataset | Lemme different = {} ",format(self.compteur))

        # for i in range(1000):
        #     logger.info("Question = ",new_games[i].questions)

        # print("question = {}".format(self.all_question))
        # exit()

        self.unique_category = np.unique(np.asarray(self.all_category))
        logger.info("Distrubution of category = {}".format(self.categories_questions))

        # logger.info(Counter(self.all_category))
        # logger.info("len=",len(self.unique_category))
        # logger.info("Conteur = ",self.compteur)

        # logger.info(self.length_question)
        # exit()
        
        super(OracleDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set,all_category=None, image_builder=None, crop_builder=None,all_img_bbox=None,all_img_describtion=None):
        
        logger.info("#################### {} ##########################".format(which_set))

        return cls(Dataset(folder, which_set, image_builder, crop_builder,all_img_bbox,all_img_describtion))




    def split(self, game):

        games = []
        all_question  = []
        all_words = []

        for i, q, a in zip(game.question_ids, game.questions, game.answers):
            
            



            # wpt = TweetTokenizer(preserve_case=False)

            new_game = copy.copy(game)
            new_game.questions = [q]
            # words = self.tokenizer.tokenize(q)
            # a = [all_words.append(word.lower()) for word in words]

            new_game.question_ids = [i]
            new_game.answers = [a]


            all_question.append(new_game.questions)

            # for all_category
            # for category in game.all_category:
            #     self.all_category.append(category)
            # last_question =  self.add_question(game.all_last_question,q,a)
            # new_game.all_last_question = last_question


            games.append(new_game)

        all_words = list(set(all_words))


        
        return games,all_words


    def add_question(self,last_questions,question,answer):


        before = ""
        i = 0

        nb_question = [i+1 for s in last_questions if s[0]!="unk"]
        list_copy = last_questions.copy()
        nb_question = int(np.sum(nb_question)) + 1

        for i in  reversed(range(6)):
            if i == 5:
                last_questions[i] = [ [answer],[question]]
            elif  nb_question > 0 and i > 0:
                last_questions[i] =  list_copy[i+1]
            nb_question -= 1

        return last_questions



class CropDataset(AbstractDataset):
    """    Each game contains no question/answers but a new object
    """
    def __init__(self, dataset):
        old_games = dataset.get_data()
        new_games = []
        for g in old_games:
            new_games += self.split(g)
        super(CropDataset, self).__init__(new_games)

    @classmethod
    def load(cls, folder, which_set, image_builder=None, crop_builder=None):
        return cls(Dataset(folder, which_set, image_builder, crop_builder))

    def split(self, game):
        games = []
        for obj in game.objects:
            new_game = copy.copy(game)
            new_game.questions = []
            new_game.question_ids = []
            new_game.answers = []
            new_game.object_id = obj.id

            # Hack the image id to differentiate objects
            new_game.image = copy.copy(game.image)
            new_game.image.id = obj.id

            games.append(new_game)

        return games


def dump_samples_into_dataset(data, save_path, tokenizer, name="model"):

    with gzip.open(save_path.format('guesswhat.' + name + '.jsonl.gz'), 'wb') as f:
        for id_game, d in enumerate(data):
            dialogue = d["dialogue"]
            game = d["game"]
            object_id = d["object_id"]
            success = d["success"]
            prob_objects = d["prob_objects"]
            guess_object_id = d["guess_object_id"]

            sample = {}

            qas = []
            start  = 1
            for k, word in enumerate(dialogue):
                if word == tokenizer.yes_token or \
                                word == tokenizer.no_token or \
                                word == tokenizer.non_applicable_token:
                    q = tokenizer.decode(dialogue[start:k - 1])
                    a = tokenizer.decode([dialogue[k]])

                    prob_obj = list(prob_objects[len(qas),:len(game.objects)])
                    prob_obj = [str(round(p,3)) for p in prob_obj] # decimal are not supported y default in json encoder

                    qas.append({"question": q,
                                "answer": a[1:-1],
                                "id":0,
                                "p": prob_obj})

                    start = k + 1

            sample["id"] = id_game
            sample["qas"] = qas
            sample["image"] = {
                "id": game.image.id,
                "width": game.image.width,
                "height": game.image.height,
                "coco_url": game.image.url
            }

            sample["objects"] = [{"id": o.id,
                                  "category_id": o.category_id,
                                  "category": o.category,
                                  "area": o.area,
                                  "bbox": o.bbox.coco_bbox,
                                  "segment" : [], #no segment to avoid making the file to big
                                  } for o in game.objects]

            sample["object_id"] = object_id
            sample["guess_object_id"] = guess_object_id
            sample["status"] = "success" if success else "failure"

            f.write(str(json.dumps(sample)).encode())
            f.write(b'\n')
